import yfinance as yf
import pandas as pd
import ta
import time
import os
import requests
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

# =========================================
# SETTINGS
# =========================================

ACCOUNT_SIZE      = 30000
RISK_PER_TRADE    = 0.01
RR_RATIO          = 2.5
MAX_ATR_STOP      = 3.0
MAX_POSITION_PCT  = 0.15          # max 15% of account per trade
SCORE_THRESHOLD   = 22
TOP_PICKS         = 5
MAX_PER_SECTOR    = 2

# Feature flags
EXPAND_UNIVERSE   = True          # Fetch S&P 500 dynamically (adds ~350 extra stocks)
NEWS_SENTIMENT    = True          # Score news headlines per pick
CHECK_15MIN       = True          # Confirm setup on 15-min chart before alerting
PORTFOLIO_FILE    = "positions.csv"
TRADE_LOG_FILE    = "trade_log.csv"

# Portfolio risk limits
MAX_PORTFOLIO_HEAT  = 0.60        # max 60% of account deployed at once
MAX_SECTOR_HEAT     = 0.20        # max 20% of account in any one sector

TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID           = os.environ.get("CHAT_ID", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"   # fast + cheap for daily scans; swap to claude-sonnet-4-6 for deeper reasoning

# =========================================
# TELEGRAM
# =========================================

def send_telegram(msg):
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        if not resp.ok:
            print(f"⚠️ Telegram error: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")

# =========================================
# BATCH DOWNLOAD HELPER
# =========================================
#
# The dynamic universe (see below) pulls in the full S&P 500 on top of the
# curated list — several hundred tickers. Downloading each one individually
# (the original approach) means several hundred separate network round trips
# per scan, which gets slower and more rate-limit-prone as the universe grows.
# This helper fetches many tickers in a handful of batched/threaded calls
# instead, and is reused for the main universe scan, market breadth, and
# sector rotation/strength checks.

def batch_download(tickers, period="2y", interval="1d", chunk_size=150):
    """
    Download OHLCV history for many tickers in as few yf.download() calls as
    possible. Returns dict {symbol: DataFrame} with flat OHLCV columns
    (dropna'd), only for symbols that returned usable data.
    """
    result  = {}
    tickers = list(dict.fromkeys(tickers))  # dedupe, preserve order
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(
                chunk, period=period, interval=interval,
                group_by="ticker", threads=True, progress=False,
            )
        except Exception as e:
            print(f"  ⚠️ batch download failed for chunk starting {chunk[0]}: {e}")
            continue
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            for sym in chunk:
                try:
                    sub = raw[sym].dropna(how="all")
                    if not sub.empty:
                        result[sym] = sub
                except Exception:
                    continue
        else:
            # yfinance collapses to single-level columns when a chunk has only 1 ticker
            sym = chunk[0]
            sub = raw.dropna(how="all")
            if not sub.empty:
                result[sym] = sub
    return result

# =========================================
# DYNAMIC UNIVERSE
# =========================================

# GICS sector name -> sector ETF code used in our system
GICS_TO_ETF = {
    "Information Technology":  "XLK",
    "Financials":               "XLF",
    "Energy":                   "XLE",
    "Health Care":              "XLV",
    "Industrials":              "XLI",
    "Utilities":                "XLU",
    "Materials":                "XLB",
    "Real Estate":              "XLRE",
    "Consumer Discretionary":   "XLY",
    "Consumer Staples":         "XLP",
    "Communication Services":   "XLC",
}

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# GitHub's raw CSV mirror of the S&P 500 constituent list — used as a fallback
# if the Wikipedia table fetch/parse fails (structure changes, blocked UA, etc.)
FALLBACK_SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

def _parse_sp500_table(df, base_sector_map, extended):
    """Shared parsing logic: df must have Symbol / GICS Sector [/ GICS Sub-Industry]."""
    added = 0
    for _, row in df.iterrows():
        sym  = str(row.get("Symbol", "")).strip().replace(".", "-")
        gics = str(row.get("GICS Sector", "")).strip()
        sub  = str(row.get("GICS Sub-Industry", "")).strip()
        if not sym or sym.lower() == "nan" or sym in extended:
            continue
        # Route semiconductors/biotech to their dedicated ETFs (matches how
        # the curated list already treats e.g. NVDA->SMH, LLY->XBI) instead
        # of lumping every new IT/Health Care name into XLK/XLV.
        if "Semiconductor" in sub:
            etf = "SMH"
        elif "Biotechnology" in sub:
            etf = "XBI"
        else:
            etf = GICS_TO_ETF.get(gics, "OTHER")
        extended[sym] = etf
        added += 1
    return added

def get_dynamic_universe(base_sector_map):
    """
    Fetch S&P 500 constituents and return an extended sector_map that combines
    the curated base_sector_map with the broader S&P 500.

    Tries Wikipedia first (with a real browser User-Agent — Wikimedia can 403
    the default urllib/pandas UA with no headers), then falls back to a GitHub-
    hosted CSV mirror of the same list if that fails for any reason. Any failure
    here should never crash the scan — worst case we fall back to the curated
    list — but we print loudly so a silent 139-only universe doesn't go unnoticed.
    """
    print("\nFetching S&P 500 universe...")
    extended = dict(base_sector_map)   # start with curated list
    before   = len(extended)

    # ---- Attempt 1: Wikipedia constituents table ----
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; swing-trading-agent-us/1.0)"}
        resp = requests.get(WIKI_SP500_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(resp.text, attrs={"id": "constituents"})
        df = tables[0]
        added = _parse_sp500_table(df, base_sector_map, extended)
        print(f"  Wikipedia OK — Curated: {before} | Added: {added} | Total: {len(extended)}")
        return extended
    except Exception as e:
        print(f"  ⚠️ Wikipedia fetch/parse failed ({type(e).__name__}: {e}) — trying fallback source...")

    # ---- Attempt 2: GitHub CSV mirror ----
    try:
        df = pd.read_csv(FALLBACK_SP500_CSV)
        df = df.rename(columns={"Sector": "GICS Sector", "Sub-Industry": "GICS Sub-Industry"})
        added = _parse_sp500_table(df, base_sector_map, extended)
        print(f"  Fallback CSV OK — Curated: {before} | Added: {added} | Total: {len(extended)}")
        return extended
    except Exception as e:
        print(f"  ⚠️ Fallback CSV also failed ({type(e).__name__}: {e}) — using curated list only ({before} stocks)")

    return extended

# =========================================
# VIX REGIME
# =========================================

def get_vix():
    try:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False)
        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        vix = float(df['Close'].iloc[-1])
        print(f"VIX: {vix:.1f}")
        return vix
    except Exception:
        return 18.0

# =========================================
# MARKET BREADTH
# =========================================

def get_market_breadth():
    sp500_sample = [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","LLY","JPM","V",
        "XOM","UNH","MA","JNJ","PG","HD","MRK","ABBV","CVX","KO",
        "PEP","AVGO","COST","WMT","BAC","CRM","TMO","ORCL","ACN","MCD",
        "CSCO","ABT","LIN","DHR","NEE","TXN","PM","WFC","UNP","RTX",
        "BMY","AMGN","QCOM","HON","INTU","IBM","GE","CAT","SPGI","ELV"
    ]
    above_50 = 0
    total    = 0
    frames = batch_download(sp500_sample, period="6mo", interval="1d")
    for sym in sp500_sample:
        try:
            df = frames.get(sym)
            if df is None or df.empty:
                continue
            df = df.dropna()
            if df.empty or len(df) < 50:
                continue
            df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50)
            if float(df['Close'].iloc[-1]) > float(df['EMA50'].iloc[-1]):
                above_50 += 1
            total += 1
        except Exception:
            continue
    if total == 0:
        return 50
    pct = round((above_50 / total) * 100, 1)
    print(f"Market Breadth: {above_50}/{total} above EMA50 = {pct}%")
    return pct

# =========================================
# SECTOR ROTATION
# =========================================

def get_sector_rotation():
    """
    Computes 1-day, 1-week and 1-month returns per sector ETF (sector momentum),
    plus a volume-trend flag used to mark "hot" sectors. Also piggybacks the
    long-term EMA200 "sector strength" check onto the same download (strength_cache
    below), so sector_is_strong() no longer needs a separate fresh download per stock.

    Returns a 3-tuple:
      hot_sectors    - set of ETF tickers flagged as currently hot (rotation bonus)
      sector_perf    - dict {etf: {"name","ret_1d","ret_1w","ret_1m","vol_trend"}} —
                       used to attach sector momentum (SectorDay/SectorWeek) to picks
      strength_cache - dict {etf: bool} — close > EMA200, i.e. long-term sector strength
    """
    sector_etfs = {
        "XLK":"Technology",   "XLF":"Financials",
        "XLE":"Energy",       "XLV":"Healthcare",
        "XLI":"Industrials",  "XLU":"Utilities",
        "XLB":"Materials",    "XLRE":"Real Estate",
        "XLY":"Consumer Disc","XLP":"Consumer Staples",
        "XLC":"Communication","SMH":"Semiconductors",
        "ITA":"Defense",      "TAN":"Solar",
        "URA":"Nuclear",      "XBI":"Biotech",
    }
    hot_sectors    = set()
    sector_perf    = {}
    strength_cache = {}

    # One batched download for all sector ETFs (1y so EMA200 strength can be derived
    # from the same data — previously a separate fresh download per stock).
    etf_frames = batch_download(list(sector_etfs.keys()), period="1y", interval="1d")

    for etf, name in sector_etfs.items():
        try:
            df = etf_frames.get(etf)
            if df is None or df.empty:
                continue
            df = df.dropna()
            if df.empty or len(df) < 21:
                continue
            ret_1d = float(df['Close'].pct_change(1).iloc[-1])
            ret_1w = float(df['Close'].pct_change(5).iloc[-1])
            ret_1m = float(df['Close'].pct_change(21).iloc[-1])
            avg_r  = float(df['Volume'].iloc[-5:].mean())
            avg_o  = float(df['Volume'].iloc[-21:-5].mean())
            vol_tr = avg_r / avg_o if avg_o > 0 else 1
            sector_perf[etf] = {
                "name": name, "ret_1d": ret_1d, "ret_1w": ret_1w,
                "ret_1m": ret_1m, "vol_trend": vol_tr
            }
            if len(df) >= 200:
                ema200 = ta.trend.ema_indicator(df['Close'], window=200)
                strength_cache[etf] = float(df['Close'].iloc[-1]) > float(ema200.iloc[-1])
            else:
                strength_cache[etf] = True
        except Exception:
            continue

    if not sector_perf:
        return hot_sectors, sector_perf, strength_cache

    all_1w = [v['ret_1w'] for v in sector_perf.values()]
    all_1m = [v['ret_1m'] for v in sector_perf.values()]
    med_1w = sorted(all_1w)[len(all_1w)//2]
    med_1m = sorted(all_1m)[len(all_1m)//2]
    print("\nSector Rotation:")
    for etf, d in sorted(sector_perf.items(), key=lambda x: x[1]['ret_1w'], reverse=True):
        is_hot = d['ret_1w'] > med_1w and d['ret_1m'] > med_1m and d['vol_trend'] > 0.9
        if is_hot:
            hot_sectors.add(etf)
        flag = "🔥" if is_hot else "  "
        print(f"  {flag} {etf:5} {d['name']:20} 1D:{d['ret_1d']:+.1%} 1W:{d['ret_1w']:+.1%} 1M:{d['ret_1m']:+.1%}")
    return hot_sectors, sector_perf, strength_cache

# =========================================
# CANDLE QUALITY
# =========================================

def candle_quality_score(df):
    score = 0
    try:
        c1 = df.iloc[-1]
        c2 = df.iloc[-2]
        c3 = df.iloc[-3]
        atr         = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else float(c1['High'] - c1['Low'])
        today_range = float(c1['High'] - c1['Low'])
        today_body  = abs(float(c1['Close'] - c1['Open']))
        today_upper = float(c1['High']) - max(float(c1['Close']), float(c1['Open']))
        close_pos   = ((float(c1['Close']) - float(c1['Low'])) / today_range) if today_range > 0 else 0.5

        if close_pos > 0.70:                                      score += 2
        elif close_pos > 0.50:                                    score += 1
        elif close_pos < 0.30:                                    score -= 2

        if today_range > 0 and today_upper / today_range > 0.40:  score -= 1
        if today_body > atr * 0.5:                                score += 1
        if float(c1['Open']) > float(c2['Close']) * 1.005:        score += 1
        if float(c1['Close']) > float(c2['High']):                score += 1

        bull = sum([
            float(c1['Close']) > float(c1['Open']),
            float(c2['Close']) > float(c2['Open']),
            float(c3['Close']) > float(c3['Open']),
        ])
        if bull == 3:    score += 1
        elif bull <= 1:  score -= 1

        if today_range > 0 and today_body / today_range < 0.10:   score -= 1  # doji
    except Exception:
        pass
    return score

# =========================================
# FUNDAMENTAL FILTER
# =========================================

SKIP_FUNDAMENTAL = {
    "SMH","SOXX","ITA","XAR","TAN","ICLN","URA","URNM",
    "ARKX","QTUM","XBI","XLU","XLI","GLD","IAU","SLV",
    "COPX","UFO","XLK","XLF","XLE","XLV","XLB","XLRE",
    "XLY","XLP","XLC","DRAM"
}

def passes_fundamental_filter(symbol):
    if symbol in SKIP_FUNDAMENTAL:
        return True
    try:
        info = yf.Ticker(symbol).info
        if not info:
            return True
        eps    = info.get("trailingEps", None)
        de     = info.get("debtToEquity", None)
        mktcap = info.get("marketCap", None)
        rev    = info.get("totalRevenue", None)
        if eps    is not None and eps    < -5:             return False
        if de     is not None and de     > 300:            return False
        if mktcap is not None and mktcap < 1_000_000_000: return False
        if rev    is not None and rev    <= 0:             return False
        return True
    except Exception:
        return True

# =========================================
# MARKET TREND
# =========================================

def market_is_bullish():
    df = yf.download("^GSPC", period="1y", interval="1d", progress=False)
    df = df.dropna()
    df.columns = df.columns.get_level_values(0)
    if df.empty:
        return False
    df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
    latest = df.iloc[-1]
    close  = float(latest['Close'])
    e50    = float(latest['EMA50'])
    e200   = float(latest['EMA200'])
    print(f"S&P: {close:.0f} | EMA50: {e50:.0f} | EMA200: {e200:.0f}")
    return close > e50 and close > e200

# =========================================
# SECTOR STRENGTH (long-term)
# =========================================

def sector_is_strong(etf_symbol):
    try:
        df = yf.download(etf_symbol, period="1y", interval="1d", progress=False)
        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if df.empty:
            return True
        df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
        return float(df['Close'].iloc[-1]) > float(df['EMA200'].iloc[-1])
    except Exception:
        return True

# =========================================
# EARNINGS FILTER
# =========================================

def is_near_earnings(symbol, days=12):
    # Note: yfinance's earnings calendar is a known weak spot — Yahoo doesn't always
    # populate it, and lookups occasionally raise (network hiccup, rate limit,
    # unexpected schema). We distinguish two cases:
    #   1. Lookup succeeds but there's genuinely no calendar data -> not a failure,
    #      don't skip.
    #   2. Lookup itself raises -> we couldn't verify earnings status at all -> FAIL
    #      CLOSED and skip the stock, rather than silently letting a possibly
    #      pre-earnings stock through.
    if symbol in SKIP_FUNDAMENTAL:
        return False
    try:
        cal = yf.Ticker(symbol).calendar
        if not cal:
            return False
        if isinstance(cal, dict):
            earn_dates = cal.get('Earnings Date', [])
            if not earn_dates:
                return False
            earn_date = pd.Timestamp(earn_dates[0]).date()
        else:
            if 'Earnings Date' not in cal.index:
                return False
            earn_date = pd.Timestamp(cal.loc['Earnings Date'].iloc[0]).date()
        diff = abs((earn_date - datetime.now().date()).days)
        if diff <= days:
            print(f"⚠️ {symbol} earnings in {diff} days — skip")
            return True
        return False
    except Exception as e:
        print(f"⚠️ {symbol}: earnings lookup failed ({e}) — skipping to be safe")
        return True

# =========================================
# NEWS SENTIMENT
# =========================================

BULLISH_WORDS = [
    "upgrade", "beat", "beats", "record", "breakout", "surge", "surges",
    "growth", "strong", "buy", "bullish", "outperform", "raises", "raised",
    "expands", "partnership", "contract", "wins", "launch", "profit",
    "revenue beat", "guidance raised", "buyback", "dividend increase"
]
BEARISH_WORDS = [
    "downgrade", "miss", "misses", "cut", "cuts", "warning", "weak",
    "loss", "losses", "sell", "probe", "fine", "recall", "investigation",
    "layoff", "layoffs", "guidance cut", "revenue miss", "bankruptcy",
    "lawsuit", "fraud", "halt", "suspended"
]

def get_news_sentiment(symbol):
    """
    Fetch recent news headlines via yfinance and score them.
    Returns (score_int, label_str, headlines_list).
    Fail-open: returns (0, 'Neutral', []) on any error.
    """
    if symbol in SKIP_FUNDAMENTAL:
        return 0, "N/A", []
    try:
        ticker = yf.Ticker(symbol)
        news   = ticker.news
        if not news:
            return 0, "Neutral", []

        score     = 0
        headlines = []
        for article in news[:6]:
            # yfinance changed its news schema to nest fields under
            # article["content"]["title"] instead of a flat article["title"].
            # Handle both shapes so this doesn't silently go blank again if
            # the schema shifts back/forward.
            title = article.get("title") or article.get("content", {}).get("title", "")
            if not title:
                continue
            low = title.lower()
            for w in BULLISH_WORDS:
                if w in low:
                    score += 1
            for w in BEARISH_WORDS:
                if w in low:
                    score -= 1
            headlines.append(title)

        if score >= 2:
            label = "Positive"
        elif score <= -2:
            label = "Negative"
        elif score == 1:
            label = "Slightly Positive"
        elif score == -1:
            label = "Slightly Negative"
        else:
            label = "Neutral"

        return score, label, headlines[:3]

    except Exception:
        return 0, "Neutral", []

# =========================================
# 15-MIN CONFIRMATION
# =========================================

def passes_15min_check(symbol, daily_entry):
    """
    Confirm the daily setup is still valid on the 15-min chart.
    Checks: price proximity, MACD direction, RSI range, volume.
    Returns (passed: bool, reason: str).
    Fail-open: returns (True, 'Data unavailable') on any error.
    """
    try:
        df = yf.download(symbol, period="5d", interval="15m", progress=False)
        if df is None or df.empty or len(df) < 30:
            return True, "Data unavailable"

        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if len(df) < 30:
            return True, "Insufficient bars"

        current_price = float(df['Close'].iloc[-1])

        # 1. Price proximity: must be within 3% of the daily entry zone
        drift = (current_price - daily_entry) / daily_entry
        if drift > 0.03:
            return False, f"Price ran +{drift:.1%} above entry — chasing risk"
        if drift < -0.04:
            return False, f"Price dropped {drift:.1%} below entry — setup breaking"

        # 2. MACD on 15-min must be bullish (above signal line)
        macd_line   = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
        macd_signal = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        macd_ok     = float(macd_line.iloc[-1]) > float(macd_signal.iloc[-1])

        # 3. RSI on 15-min: momentum zone 45-78
        rsi    = ta.momentum.rsi(df['Close'], window=14)
        rsi_v  = float(rsi.iloc[-1])
        rsi_ok = 45 < rsi_v < 78

        # 4. Recent volume above its 20-bar average on 15-min
        avg_vol = float(df['Volume'].iloc[-20:].mean())
        cur_vol = float(df['Volume'].iloc[-3:].mean())   # last 45 min
        vol_ok  = cur_vol >= avg_vol * 0.8               # at least 80% of avg

        fails = []
        if not macd_ok:  fails.append(f"15m MACD bearish")
        if not rsi_ok:   fails.append(f"15m RSI={rsi_v:.0f}")
        if not vol_ok:   fails.append(f"15m vol thin ({cur_vol/avg_vol:.0%} avg)")

        if len(fails) >= 2:
            return False, " | ".join(fails)
        elif fails:
            return True, f"⚠️ Minor: {fails[0]}"
        else:
            return True, f"✅ RSI={rsi_v:.0f}, MACD bullish, vol OK"

    except Exception as e:
        return True, f"Check skipped ({e})"

# =========================================
# PORTFOLIO RISK
# =========================================

def get_portfolio_positions():
    """
    Read current open positions from positions.csv.
    Format: symbol, shares, entry_price, sector
    Returns dict keyed by symbol.
    """
    pos_file = Path(PORTFOLIO_FILE)
    if not pos_file.exists():
        return {}

    positions = {}
    try:
        with open(pos_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = row.get('symbol', '').strip().upper()
                if not sym:
                    continue
                try:
                    positions[sym] = {
                        'shares':  int(float(row.get('shares', 0))),
                        'entry':   float(row.get('entry_price', 0)),
                        'sector':  row.get('sector', 'OTHER').strip(),
                    }
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"⚠️ Could not read {PORTFOLIO_FILE}: {e}")

    return positions

def get_portfolio_heat(positions):
    """
    Calculate total deployed capital and per-sector exposure.
    Returns (total_pct, sector_pct_dict, summary_str).
    """
    if not positions:
        return 0.0, {}, "No open positions"

    total_invested = 0.0
    sector_invested = {}

    for sym, pos in positions.items():
        value = pos['shares'] * pos['entry']
        total_invested += value
        sec = pos['sector']
        sector_invested[sec] = sector_invested.get(sec, 0.0) + value

    total_pct  = total_invested / ACCOUNT_SIZE
    sector_pct = {sec: v / ACCOUNT_SIZE for sec, v in sector_invested.items()}

    lines = [f"Portfolio heat: {total_pct:.0%} deployed (${total_invested:,.0f})"]
    for sec, pct in sorted(sector_pct.items(), key=lambda x: -x[1]):
        bar = "🔴" if pct > MAX_SECTOR_HEAT else "🟡" if pct > MAX_SECTOR_HEAT * 0.7 else "🟢"
        lines.append(f"  {bar} {sec}: {pct:.0%}")

    return total_pct, sector_pct, "\n".join(lines)

def pick_blocked_by_portfolio(pick, positions, sector_pct):
    """
    Return (blocked: bool, reason: str).
    Blocks if: symbol already held, sector over limit, or total heat too high.
    """
    sym    = pick['Symbol']
    sec    = pick['Sector']
    invest = pick['Invested']

    # Already holding this symbol
    if sym in positions:
        return True, f"{sym} already in portfolio"

    # Adding this pick would push sector over limit
    cur_sec_pct  = sector_pct.get(sec, 0.0)
    new_sec_pct  = cur_sec_pct + (invest / ACCOUNT_SIZE)
    if new_sec_pct > MAX_SECTOR_HEAT:
        return True, f"{sec} sector would be {new_sec_pct:.0%} > {MAX_SECTOR_HEAT:.0%} limit"

    return False, ""

# =========================================
# TRADE LOGGING
# =========================================

TRADE_LOG_FIELDS = [
    'date', 'symbol', 'sector', 'setup', 'score',
    'entry', 'stop', 'target', 'size', 'invested',
    'risk_usd', 'reward_usd', 'rr',
    'news_sentiment', 'confirmed_15m',
    'outcome', 'outcome_date', 'exit_price', 'pnl_usd', 'pnl_pct'
]

def log_picks(picks, confirmed_map, sentiment_map):
    """Append today's picks to trade_log.csv (won't duplicate same symbol+date)."""
    log_file   = Path(TRADE_LOG_FILE)
    today_str  = datetime.now().strftime('%Y-%m-%d')
    file_exists = log_file.exists()

    # Load existing entries to avoid duplicates
    existing = set()
    if file_exists:
        try:
            with open(log_file, newline='') as f:
                for row in csv.DictReader(f):
                    existing.add((row.get('date',''), row.get('symbol','')))
        except Exception:
            pass

    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()

        for pick in picks:
            sym = pick['Symbol']
            if (today_str, sym) in existing:
                continue  # already logged today
            rr = round(pick['Reward$'] / pick['Risk$'], 2) if pick['Risk$'] > 0 else 0
            writer.writerow({
                'date':           today_str,
                'symbol':         sym,
                'sector':         pick['Sector'],
                'setup':          pick['Setup'],
                'score':          pick['Score'],
                'entry':          pick['Entry'],
                'stop':           pick['Stop'],
                'target':         pick['Target'],
                'size':           pick['Size'],
                'invested':       int(pick['Invested']),
                'risk_usd':       int(pick['Risk$']),
                'reward_usd':     int(pick['Reward$']),
                'rr':             rr,
                'news_sentiment': sentiment_map.get(sym, ('', 'N/A', []))[1],
                'confirmed_15m':  'Yes' if confirmed_map.get(sym, (True,''))[0] else 'No',
                'outcome':        '',
                'outcome_date':   '',
                'exit_price':     '',
                'pnl_usd':        '',
                'pnl_pct':        '',
            })

    print(f"📋 Logged {len(picks)} picks to {TRADE_LOG_FILE}")

def update_trade_outcomes():
    """
    For every open trade in trade_log.csv (outcome == ''),
    fetch the current price and check if target or stop has been hit.
    Fills in outcome, exit_price, pnl_usd, pnl_pct, outcome_date.
    """
    log_file = Path(TRADE_LOG_FILE)
    if not log_file.exists():
        return

    rows     = []
    updated  = 0
    today_str = datetime.now().strftime('%Y-%m-%d')

    try:
        with open(log_file, newline='') as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"⚠️ Could not read trade log: {e}")
        return

    for row in rows:
        if row.get('outcome', '').strip():
            continue   # already closed

        sym    = row.get('symbol', '')
        entry  = float(row.get('entry', 0) or 0)
        stop   = float(row.get('stop', 0) or 0)
        target = float(row.get('target', 0) or 0)
        size   = int(float(row.get('size', 0) or 0))

        if not sym or entry <= 0:
            continue

        try:
            df = yf.download(sym, period="5d", interval="1d", progress=False)
            if df is None or df.empty:
                continue
            df = df.dropna()
            df.columns = df.columns.get_level_values(0)

            # Check high/low of last bar to see if stop or target was reached
            last   = df.iloc[-1]
            hi     = float(last['High'])
            lo     = float(last['Low'])
            close  = float(last['Close'])

            outcome    = ''
            exit_price = close

            if lo <= stop:
                outcome    = 'STOPPED'
                exit_price = stop
            elif hi >= target:
                outcome    = 'TARGET HIT'
                exit_price = target

            if outcome:
                pnl_usd = round((exit_price - entry) * size, 2)
                pnl_pct = round((exit_price - entry) / entry * 100, 2)
                row['outcome']      = outcome
                row['outcome_date'] = today_str
                row['exit_price']   = exit_price
                row['pnl_usd']      = pnl_usd
                row['pnl_pct']      = pnl_pct
                updated += 1
                emoji = "✅" if outcome == 'TARGET HIT' else "❌"
                print(f"  {emoji} {sym}: {outcome} | P&L ${pnl_usd:+.0f} ({pnl_pct:+.1f}%)")
            else:
                # Still open — update unrealized P&L
                unreal = round((close - entry) * size, 2)
                unreal_pct = round((close - entry) / entry * 100, 2)
                print(f"  🔄 {sym}: open | price ${close:.2f} | unrealized ${unreal:+.0f} ({unreal_pct:+.1f}%)")

        except Exception as e:
            print(f"  ⚠️ {sym} outcome check failed: {e}")
            continue

    if updated > 0:
        # Write all rows back (with updated outcomes)
        try:
            with open(log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"📋 Updated {updated} trade outcome(s) in {TRADE_LOG_FILE}")
        except Exception as e:
            print(f"⚠️ Could not write trade log: {e}")

def print_trade_stats():
    """Print a simple win-rate / P&L summary from closed trades."""
    log_file = Path(TRADE_LOG_FILE)
    if not log_file.exists():
        return

    try:
        with open(log_file, newline='') as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return

    closed = [r for r in rows if r.get('outcome', '').strip() in ('TARGET HIT', 'STOPPED')]
    if not closed:
        return

    wins  = [r for r in closed if r['outcome'] == 'TARGET HIT']
    total = len(closed)
    win_r = len(wins) / total * 100

    try:
        pnls  = [float(r['pnl_usd']) for r in closed if r.get('pnl_usd')]
        net   = sum(pnls)
        avg   = net / len(pnls) if pnls else 0
        print(f"\n📈 Trade History: {total} closed | Win rate: {win_r:.0f}% | Net P&L: ${net:+,.0f} | Avg: ${avg:+.0f}")
    except Exception:
        print(f"\n📈 Trade History: {total} closed | Win rate: {win_r:.0f}%")

# =========================================
# MAIN TECHNICAL SCANNER
# =========================================

def check_stock(symbol, df, spy_df, hot_sectors, sector_perf, active_sector_map, risk_pct=RISK_PER_TRADE):
    # `df` is now a pre-fetched, per-symbol OHLCV frame sliced out of a single batched
    # universe download (see batch_download()/run_agent()) instead of a fresh
    # individual yf.download() call per stock — the main perf change for the
    # dynamic (S&P 500-sized) universe.
    try:
        if df is None or df.empty or len(df) < 250:
            return None
        df = df.dropna()
        if df.empty or len(df) < 250:
            return None

        price = float(df['Close'].iloc[-1])
        if price < 5.0:
            return None

        avg_dv = float(df['Close'].iloc[-20:].mean() * df['Volume'].iloc[-20:].mean())
        if avg_dv < 2_000_000:
            return None

        # ---- Indicators ----
        df['EMA10']  = ta.trend.ema_indicator(df['Close'], window=10)
        df['EMA20']  = ta.trend.ema_indicator(df['Close'], window=20)
        df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
        df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
        df['RSI']    = ta.momentum.rsi(df['Close'], window=14)
        df['AvgVol'] = df['Volume'].rolling(20).mean()
        df['HH20']   = df['High'].rolling(20).max()
        df['High52'] = df['High'].rolling(252).max()

        df['TR'] = (
            df['High'] - df['Low']
        ).combine(abs(df['High'] - df['Close'].shift(1)), max
        ).combine(abs(df['Low']  - df['Close'].shift(1)), max)
        df['ATR'] = df['TR'].rolling(14).mean()

        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        adx_val   = float(df['ADX'].iloc[-1]) if not pd.isna(df['ADX'].iloc[-1]) else 20.0

        df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
        df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']
        bb_squeeze = False
        if len(df) >= 60:
            pct20      = float(df['BB_width'].iloc[-60:].quantile(0.20))
            bb_squeeze = float(df['BB_width'].iloc[-1]) < pct20

        df['OBV']       = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['OBV_EMA20'] = ta.trend.ema_indicator(df['OBV'], window=20)
        obv_rising      = float(df['OBV'].iloc[-1]) > float(df['OBV_EMA20'].iloc[-1])
        obv_slope       = float(df['OBV'].iloc[-1]) - float(df['OBV'].iloc[-10])
        obv_trend_pos   = obv_slope > 0

        macd_line   = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
        macd_signal = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        macd_hist   = ta.trend.macd_diff(df['Close'], window_slow=26, window_fast=12, window_sign=9)

        macd_now      = float(macd_line.iloc[-1])
        macd_sig_now  = float(macd_signal.iloc[-1])
        macd_prev     = float(macd_line.iloc[-2])
        macd_sig_prev = float(macd_signal.iloc[-2])
        macd_hist_now = float(macd_hist.iloc[-1])
        macd_hist_prv = float(macd_hist.iloc[-2])

        macd_crossed_up  = macd_prev < macd_sig_prev and macd_now > macd_sig_now
        macd_above_sig   = macd_now > macd_sig_now
        macd_hist_rising = macd_hist_now > macd_hist_prv and macd_hist_now > 0

        s3  = df['Close'].pct_change(63).iloc[-1]
        s6  = df['Close'].pct_change(126).iloc[-1]
        s12 = df['Close'].pct_change(252).iloc[-1]
        n3  = spy_df['Close'].pct_change(63).iloc[-1]
        n6  = spy_df['Close'].pct_change(126).iloc[-1]
        n12 = spy_df['Close'].pct_change(252).iloc[-1]
        rs  = sum([s3 > n3, s6 > n6, s12 > n12])

        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        score  = 0

        if rs >= 2:  score += 2
        if rs == 3:  score += 1

        if latest['EMA10']  > latest['EMA20']:  score += 2
        if latest['EMA20']  > latest['EMA50']:  score += 2
        if latest['EMA50']  > latest['EMA200']: score += 2
        if latest['Close']  > latest['EMA50']:  score += 1
        if latest['Close']  > latest['EMA200']: score += 2

        rsi      = float(latest['RSI'])
        rsi_prev = float(prev['RSI'])
        if rsi > 60 and rsi_prev < 60:         score += 2
        elif 55 < rsi < 75:                     score += 2
        elif 40 < rsi < 55 and rsi > rsi_prev: score += 1
        elif rsi > 80:                          score -= 2
        elif rsi < 40:                          score -= 1

        if latest['Close'] > prev['HH20']:  score += 2
        ath_dist = latest['Close'] / latest['High52']
        if ath_dist > 0.90:                 score += 2
        elif ath_dist > 0.80:              score += 1

        rvol = 0.0
        if latest['AvgVol'] > 0:
            rvol = latest['Volume'] / latest['AvgVol']
            if rvol > 2.0:    score += 2
            elif rvol > 1.5:  score += 1

        if latest['ATR'] > df['ATR'].iloc[-5]:  score += 1

        dist = (latest['Close'] - latest['EMA20']) / latest['EMA20']
        if dist < 0.05:    score += 2
        elif dist < 0.08:  score += 1
        elif dist > 0.20:  score -= 1

        if len(df) >= 60 and len(spy_df) >= 60:
            sr = float(df['Close'].squeeze().pct_change(60).iloc[-1])
            nr = float(spy_df['Close'].squeeze().pct_change(60).iloc[-1])
            if sr > nr * 1.5:  score += 2
            elif sr > nr:      score += 1

        recent_high = df['High'].iloc[-10:].max()
        if latest['Close'] < recent_high * 0.85:  score -= 2

        # ---- OVEREXTENSION / "STRAIGHT UP, NEEDS COOL-OFF" ----
        # Mirrors the India agent's check: catches the case a human reviewer
        # flags on sight ("looks nice but straight up here, would be better
        # with 1-2 days cool off") that the rest of the scorer actually
        # rewards — fresh RSI cross, rising MACD hist, rising OBV, and a
        # volume breakout all fire on exactly this kind of sharp, un-rested
        # move. This looks at the last 3 sessions directly instead of
        # relying on those proxies.
        last3 = df.iloc[-3:]
        up_days_3 = int((last3['Close'] > last3['Open']).sum())
        move_3d = float((latest['Close'] - df['Close'].iloc[-4]) / df['Close'].iloc[-4]) if len(df) >= 4 else 0.0
        no_pullback_1d = float(latest['Close']) > float(prev['Close']) and float(prev['Close']) > float(df['Close'].iloc[-3])

        overextended = False
        if move_3d > 0.08 and up_days_3 >= 2:
            overextended = True
        if up_days_3 == 3 and move_3d > 0.06:
            overextended = True

        # Institutional breakout exception: when volume is 3x+ normal AND stock is
        # near/at its 52-week high, the move is likely real institutional buying — not
        # retail chasing. Applying the overextension penalty here would have filtered
        # out CRM/BE-style gap-and-hold breakouts. Skip the penalty in this case.
        is_institutional_breakout = rvol > 3.0 and ath_dist > 0.92

        if not is_institutional_breakout:
            if overextended:
                score -= 3
            elif no_pullback_1d and move_3d > 0.05:
                score -= 1

        if obv_rising and obv_trend_pos:               score += 2
        elif obv_rising:                                score += 1
        elif not obv_trend_pos and not obv_rising:      score -= 2

        if macd_crossed_up:                             score += 2
        elif macd_above_sig and macd_hist_rising:       score += 2
        elif macd_above_sig:                            score += 1
        elif not macd_above_sig and macd_hist_now < 0: score -= 1

        if adx_val > 30:    score += 2
        elif adx_val > 20:  score += 1
        elif adx_val < 15:  score -= 2

        if bb_squeeze:  score += 3

        candle_score = candle_quality_score(df)
        score += candle_score

        stock_sector = active_sector_map.get(symbol, "OTHER")
        if stock_sector in hot_sectors:  score += 2

        # ---- SETUP TYPE (determined before threshold so we can adjust it) ----
        if bb_squeeze and latest['Close'] > prev['HH20']:
            setup_type = "Squeeze Breakout"
        elif latest['Close'] > prev['HH20'] and rvol > 2.0:
            setup_type = "Volume Breakout"
        elif dist < 0.05 and rsi > 50:
            setup_type = "EMA20 Pullback"
        elif ath_dist > 0.95:
            setup_type = "ATH Breakout"
        elif bb_squeeze:
            setup_type = "Squeeze Setup"
        else:
            setup_type = "Trend Continuation"

        # Breakout setups (Volume, ATH, Squeeze) already embed hard price/volume
        # evidence in their definition — they don't need as many confirmatory points
        # from the EMA-stack and RS scoring to be valid. Lowering their bar by 4
        # points (22 → 18) catches early-stage breakouts that haven't yet built a
        # clean EMA stack but have the volume and price proof.
        if setup_type in ("Volume Breakout", "ATH Breakout", "Squeeze Breakout"):
            effective_threshold = SCORE_THRESHOLD - 4
        else:
            effective_threshold = SCORE_THRESHOLD

        # Momentum alert: high-volume breakout near 52w high with a minimum sanity
        # score — flagged even if it doesn't reach the effective_threshold. Sent as
        # a separate "⚡ MOMENTUM ALERT" so the user knows it's higher risk.
        is_momentum_alert = (
            rvol > 2.5 and
            float(latest['Close']) > float(prev['HH20']) and
            ath_dist > 0.90 and
            float(latest['Close']) > float(latest['EMA50']) and
            score >= 12
        )

        if score < effective_threshold and not is_momentum_alert:
            return None

        # ---- POSITION SIZING ----
        entry          = float(latest['Close'])
        atr            = float(latest['ATR'])
        ten_bar_low    = float(df['Low'].iloc[-10:].min())
        atr_stop       = entry - (MAX_ATR_STOP * atr)
        stop           = max(ten_bar_low, atr_stop)
        risk           = entry - stop

        if risk <= 0 or risk > entry * 0.12:
            return None

        max_by_dollars = int((ACCOUNT_SIZE * MAX_POSITION_PCT) / entry)
        size           = min(int((ACCOUNT_SIZE * risk_pct) / risk), max_by_dollars)
        if size <= 0:
            return None

        target   = entry + (risk * RR_RATIO)
        invested = round(entry * size, 0)
        acct_pct = round((invested / ACCOUNT_SIZE) * 100, 1)

        if candle_score >= 3:    candle_label = "Strong"
        elif candle_score >= 1:  candle_label = "Good"
        elif candle_score == 0:  candle_label = "Neutral"
        else:                    candle_label = "Weak"

        if macd_crossed_up:   macd_label = "Fresh cross"
        elif macd_above_sig:  macd_label = "Bullish"
        else:                  macd_label = "Bearish"

        obv_label = "Confirming" if obv_rising and obv_trend_pos else \
                    "Rising"     if obv_rising else "Diverging"

        adx_label = f"{adx_val:.0f} ({'Strong' if adx_val > 30 else 'Moderate' if adx_val > 20 else 'Weak'})"

        if overextended:
            extension_label = f"⚠️ Extended ({move_3d:+.1%}/3d, {up_days_3}/3 up) — consider 1-2 day cool-off"
        elif no_pullback_1d and move_3d > 0.05:
            extension_label = f"Slightly stretched ({move_3d:+.1%}/3d)"
        else:
            extension_label = "OK"

        # ---- SECTOR MOMENTUM (day/week) for this pick's sector ----
        sp = sector_perf.get(stock_sector, {}) if sector_perf else {}
        sector_day_str  = f"{sp['ret_1d']:+.1%}" if 'ret_1d' in sp else "n/a"
        sector_week_str = f"{sp['ret_1w']:+.1%}" if 'ret_1w' in sp else "n/a"

        return {
            "Symbol":     symbol,
            "Sector":     stock_sector,
            "Score":      score,
            "Setup":      setup_type,
            "Candle":     candle_label,
            "MACD":       macd_label,
            "OBV":        obv_label,
            "ADX":        adx_label,
            "SectorDay":  sector_day_str,
            "SectorWeek": sector_week_str,
            "Squeeze":    "Yes" if bb_squeeze else "No",
            "Extension":  extension_label,
            "RVOL":       round(rvol, 1),
            "RSI":        round(rsi, 1),
            "Move3d":     f"{move_3d:+.1%}",
            "MomentumAlert": is_momentum_alert and score < effective_threshold,
            "Entry":    round(entry, 2),
            "Stop":     round(float(stop), 2),
            "Target":   round(float(target), 2),
            "Size":     size,
            "Invested": invested,
            "AcctPct":  acct_pct,
            "Risk$":    round(risk * size, 0),
            "Reward$":  round((float(target) - entry) * size, 0),
        }

    except Exception as e:
        print(f"  ⚠️ {symbol}: {e}")
        return None

# =========================================
# STOCK UNIVERSE (curated base)
# =========================================

sector_map = {
    # Semiconductors
    "NVDA":"SMH",  "AMD":"SMH",   "AVGO":"SMH",  "TSM":"SMH",
    "AMAT":"SMH",  "LRCX":"SMH",  "KLAC":"SMH",  "ASML":"SMH",
    "ARM":"SMH",   "MRVL":"SMH",  "ONTO":"SMH",  "ENTG":"SMH",
    "SMH":"SMH",   "SOXX":"SOXX", "SMCI":"SMH",
    "SNDK":"SMH",  "WDC":"SMH",   "MU":"SMH",    "STX":"SMH",
    "DRAM":"SMH",
    # Technology / Software
    "PLTR":"XLK",  "MSFT":"XLK",  "AMZN":"XLK",  "NOW":"XLK",
    "CRM":"XLK",   "ADBE":"XLK",  "INTU":"XLK",  "APP":"XLK",
    "CRWD":"XLK",  "PANW":"XLK",  "ZS":"XLK",    "FTNT":"XLK",
    "OKTA":"XLK",  "S":"XLK",     "CYBR":"XLK",  "SNOW":"XLK",
    "DDOG":"XLK",  "NET":"XLK",   "ORCL":"XLK",  "MDB":"XLK",
    "GTLB":"XLK",  "ANET":"XLK",  "DELL":"XLK",  "HPE":"XLK",
    "AXON":"XLK",  "CORT":"XLK",  "IONQ":"XLK",  "RGTI":"XLK",
    "QUBT":"XLK",  "ADSK":"XLK",  "SHOP":"XLK",  "MNDY":"XLK",
    "GDDY":"XLK",
    # Communication
    "GOOGL":"XLC", "META":"XLC",  "RBLX":"XLC",  "NFLX":"XLC",
    "DUOL":"XLC",
    # Consumer Discretionary
    "TSLA":"XLY",  "UBER":"XLY",  "LYFT":"XLY",  "DECK":"XLY",
    "ONON":"XLY",  "LULU":"XLY",  "MELI":"XLY",  "SE":"XLY",
    "EXPE":"XLY",
    # Utilities / Power
    "NEE":"XLU",   "ICLN":"XLU",  "VST":"XLU",   "CEG":"XLU",
    "NRG":"XLU",
    # Solar
    "FSLR":"TAN",  "ENPH":"TAN",  "SEDG":"TAN",  "TAN":"TAN",
    # Industrials
    "VRT":"XLI",   "ETN":"XLI",   "GEV":"XLI",   "PWR":"XLI",
    "ACHR":"XLI",  "GE":"XLI",    "BA":"XLI",
    # Real Estate
    "EQIX":"XLRE", "DLR":"XLRE",  "AMT":"XLRE",
    # Defense
    "LMT":"ITA",   "RTX":"ITA",   "NOC":"ITA",   "GD":"ITA",
    "KTOS":"ITA",  "LDOS":"ITA",  "HII":"ITA",   "TDG":"ITA",
    "ITA":"ITA",   "XAR":"XAR",   "RKLB":"ITA",  "ASTS":"ITA",
    "LUNR":"ITA",
    # Nuclear
    "CCJ":"URA",   "NXE":"URA",   "LEU":"URA",   "SMR":"URA",
    "OKLO":"URA",  "URA":"URA",   "URNM":"URA",
    # Biotech / Healthcare
    "LLY":"XBI",   "NVO":"XBI",   "VKTX":"XBI",  "RXRX":"XBI",
    "ROIV":"XBI",  "XBI":"XBI",
    # Financials
    "GS":"XLF",    "JPM":"XLF",   "V":"XLF",     "MA":"XLF",
    "PYPL":"XLF",  "AFRM":"XLF",  "HOOD":"XLF",  "IBKR":"XLF",
    "SOFI":"XLF",  "NU":"XLF",    "COF":"XLF",   "COIN":"XLF",
    # Consumer Staples
    "COST":"XLP",  "WMT":"XLP",   "CELH":"XLP",
    # Energy
    "XOM":"XLE",   "CVX":"XLE",   "MPC":"XLE",
    # Commodities / Materials
    "GLD":"GLD",   "IAU":"GLD",   "SLV":"SLV",
    "COPX":"XLB",  "WPM":"XLB",   "GOLD":"XLB",  "MP":"XLB",
    "ALB":"XLB",   "SQM":"XLB",
}

# =========================================
# SIGNAL PERFORMANCE (reads trade_log.csv)
# =========================================

def get_recent_performance(n_days: int = 60) -> dict:
    """
    Read trade_log.csv and compute signal quality stats for the last n_days.

    Logs every signal the bot generated — not only the trades the user took.
    This measures the screener's accuracy, which Claude uses to weight setups
    (e.g. "ATH Breakout has 70% win rate lately, EMA Pullback only 40%").
    """
    path = Path(TRADE_LOG_FILE)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty or "outcome" not in df.columns:
            return {}

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        cutoff  = datetime.now() - timedelta(days=n_days)
        recent  = df[df["date"] >= cutoff].copy()
        closed  = recent[recent["outcome"].isin(["TARGET HIT", "STOPPED"])].copy()

        def stats(rows):
            total = len(rows)
            wins  = int((rows["outcome"] == "TARGET HIT").sum())
            return {
                "trades":   total,
                "wins":     wins,
                "win_rate": round(wins / total, 2) if total > 0 else None,
            }

        by_setup  = {g: stats(gdf) for g, gdf in closed.groupby("setup")  if len(gdf) >= 2}
        by_sector = {g: stats(gdf) for g, gdf in closed.groupby("sector") if len(gdf) >= 2}

        last5  = closed.sort_values("outcome_date").tail(5)
        streak = " → ".join(
            "✅W" if r == "TARGET HIT" else "❌L"
            for r in last5["outcome"]
        ) if not last5.empty else "No closed trades yet"

        return {
            "window_days":    n_days,
            "total_signals":  len(recent),
            "overall":        stats(closed),
            "by_setup":       by_setup,
            "by_sector":      by_sector,
            "recent_streak":  streak,
        }
    except Exception as e:
        print(f"  ⚠️ Performance read failed: {e}")
        return {}


# =========================================
# CLAUDE REASONING LAYER
# =========================================

def claude_reason(candidates: list, market_ctx: dict, perf: dict) -> dict:
    """
    Send candidates + market context + recent signal performance to Claude.
    Claude ranks the picks by conviction, flags ones to skip, and gives a
    one-paragraph market read. Returns a parsed dict; empty dict on failure.

    The user won't take every alert — Claude's job is to help them prioritise
    and to explain the 'why' using actual recent win-rate data from trade_log.
    """
    if not ANTHROPIC_API_KEY:
        print("  ⚠️ ANTHROPIC_API_KEY not set — skipping Claude reasoning")
        return {}

    try:
        import anthropic
    except ImportError:
        print("  ⚠️ anthropic package not installed — pip install anthropic")
        return {}

    # ---- Build performance context string ----
    perf_lines = []
    if perf:
        overall = perf.get("overall", {})
        wr      = overall.get("win_rate")
        n       = overall.get("trades", 0)
        perf_lines.append(
            f"Overall win rate (last {perf.get('window_days', 60)}d): "
            + (f"{int(wr*100)}% across {n} closed signals" if wr is not None else "Not enough data yet")
        )
        by_setup = perf.get("by_setup", {})
        if by_setup:
            perf_lines.append("Win rate by setup type:")
            for s, v in sorted(by_setup.items(), key=lambda x: -(x[1].get("win_rate") or 0)):
                perf_lines.append(f"  {s}: {int(v['win_rate']*100)}% ({v['wins']}/{v['trades']})")
        by_sector = perf.get("by_sector", {})
        if by_sector:
            perf_lines.append("Win rate by sector:")
            for s, v in sorted(by_sector.items(), key=lambda x: -(x[1].get("win_rate") or 0)):
                perf_lines.append(f"  {s}: {int(v['win_rate']*100)}% ({v['wins']}/{v['trades']})")
        perf_lines.append(f"Recent streak (last 5 closed): {perf.get('recent_streak', 'N/A')}")

    # ---- Build candidates string ----
    cand_lines = []
    for i, c in enumerate(candidates, 1):
        cand_lines.append(
            f"{i}. {c['Symbol']} [{c['Sector']}] | {c['Setup']} | "
            f"Score:{c['Score']} RVOL:{c['RVOL']}x RSI:{c['RSI']} "
            f"MACD:{c['MACD']} OBV:{c['OBV']} ADX:{c['ADX']} "
            f"Entry:${c['Entry']} Stop:${c['Stop']} Target:${c['Target']} "
            f"Extension:{c['Extension']} "
            f"Sector 1D:{c['SectorDay']} 1W:{c['SectorWeek']}"
        )

    prompt = f"""You are a professional swing trading analyst reviewing today's scan for US stocks.
The user will review these alerts and decide which trades to actually take — they do NOT enter every signal.
Your job: help them prioritise and explain the WHY using the performance data below.

MARKET CONDITIONS:
- S&P 500  : {market_ctx.get('regime', 'Bullish')}
- VIX      : {market_ctx.get('vix_label', 'N/A')}
- Breadth  : {market_ctx.get('breadth_label', 'N/A')} ({market_ctx.get('breadth', 'N/A')}% above EMA50)
- Hot sectors: {market_ctx.get('hot_str', 'None')}

RECENT SIGNAL PERFORMANCE (measures screener accuracy, not user P&L):
{chr(10).join(perf_lines) if perf_lines else 'No performance data yet — first scan.'}

TODAY'S CANDIDATES ({len(candidates)} stocks, ranked by score):
{chr(10).join(cand_lines)}

Respond ONLY with a valid JSON object — no markdown fences, no extra text:
{{
  "market_read": "<2 sentences: current market tone and what it means for swing trades today>",
  "picks": [
    {{"symbol": "TICKER", "conviction": "high|medium|low",
      "reason": "<1-2 sentences referencing indicators AND recent performance data>"}}
  ],
  "skip": [
    {{"symbol": "TICKER", "reason": "<1 sentence why you'd pass on this one>"}}
  ],
  "overall_confidence": "high|medium|low",
  "confidence_reason": "<one sentence>"
}}

Rules:
- picks: select the top 3-5 by conviction. Downweight setups/sectors with poor recent win rate.
- skip: list any candidates you'd avoid (sector exhaustion, poor setup history, overextension).
- Be specific — cite actual numbers from the indicator data and performance stats.
- A setup with 70%+ win rate recently deserves higher conviction than one at 40%.
"""

    try:
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model      = CLAUDE_MODEL,
            max_tokens = 1024,
            messages   = [{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Strip accidental markdown fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  ⚠️ Claude JSON parse error: {e}")
        return {}
    except Exception as e:
        print(f"  ⚠️ Claude API error: {e}")
        return {}


# =========================================
# TOP MOVERS INJECTION
# =========================================

def get_top_movers(n: int = 40) -> list:
    """
    Fetch today's top-gaining US stocks from the Yahoo Finance day-gainers
    screener. Returns a list of ticker symbols that gained >3% on the day
    and aren't already in the curated universe — injects them so the scanner
    can catch breakouts that aren't in the S&P 500 or the hardcoded list.
    """
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params  = {"scrIds": "day_gainers", "count": n, "formatted": "false"}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if not r.ok:
            print(f"  ⚠️ Top movers fetch HTTP {r.status_code}")
            return []
        quotes = r.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
        movers = [
            q["symbol"] for q in quotes
            if q.get("regularMarketChangePercent", 0) > 3.0
            and q.get("regularMarketVolume", 0) > 500_000
            and "." not in q.get("symbol", ".")   # skip ADRs like BRK.B
        ]
        print(f"  📈 Top movers today: {movers[:20]}")
        return movers
    except Exception as e:
        print(f"  ⚠️ Top movers fetch failed: {e}")
        return []


# =========================================
# MAIN
# =========================================

def run_agent():

    print(f"\n{'='*55}")
    print(f"US Pro Scan — {datetime.now().strftime('%d %b %Y %H:%M:%S')}")
    print(f"{'='*55}")

    # ---- Step 0: Check open trade outcomes from prior runs ----
    print("\nChecking open trade outcomes...")
    update_trade_outcomes()
    print_trade_stats()

    # ---- Step 1: Market regime filters ----
    if not market_is_bullish():
        msg = (
            f"📉 S&P below EMA50/EMA200 — cash only\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')}"
        )
        send_telegram(msg)
        return

    vix = get_vix()
    if vix > 30:
        send_telegram(
            f"⚠️ VIX={vix:.0f} — extreme fear mode\n"
            f"Swing setups fail at high rates when VIX >30.\n"
            f"Skipping scan — stay in cash.\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')}"
        )
        return

    effective_risk = RISK_PER_TRADE * (0.5 if vix > 25 else 1.0)
    vix_label      = f"Elevated ({vix:.0f}) — half size" if vix > 25 else f"Normal ({vix:.0f})"

    print("\nChecking market breadth...")
    breadth = get_market_breadth()
    if breadth < 40:
        msg = (
            f"⚠️ Market breadth WEAK ({breadth}%)\n"
            f"Only {breadth}% of S&P stocks healthy.\n"
            f"Skipping — too risky.\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')}"
        )
        send_telegram(msg)
        return

    breadth_label = "Strong" if breadth >= 60 else "Mixed" if breadth >= 40 else "Weak"

    print("\nChecking sector rotation...")
    hot_sectors, sector_perf, sector_strength_cache = get_sector_rotation()

    # ---- Step 2: Build universe ----
    active_sector_map = get_dynamic_universe(sector_map) if EXPAND_UNIVERSE else dict(sector_map)

    # Inject today's top movers so stocks outside the S&P 500 or curated list
    # still get scanned on their breakout day (e.g. mid-caps, recent IPOs).
    print("\nFetching today's top movers...")
    for sym in get_top_movers(40):
        if sym not in active_sector_map:
            active_sector_map[sym] = "OTHER"

    all_stocks = list(dict.fromkeys(active_sector_map.keys()))

    # ---- Step 3: Portfolio risk snapshot ----
    positions          = get_portfolio_positions()
    total_heat, sector_pct, heat_summary = get_portfolio_heat(positions)
    print(f"\n{heat_summary}")

    if total_heat >= MAX_PORTFOLIO_HEAT:
        msg = (
            f"🔴 Portfolio fully deployed ({total_heat:.0%}) — no new entries\n"
            f"Max heat: {MAX_PORTFOLIO_HEAT:.0%}\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')}"
        )
        send_telegram(msg)
        return

    # ---- Step 4: SPY reference ----
    spy_df = yf.download("^GSPC", period="1y", interval="1d", progress=False)
    if spy_df is None or spy_df.empty:
        return
    spy_df = spy_df.dropna()
    spy_df.columns = spy_df.columns.get_level_values(0)

    # ---- Step 5: Scan ----
    picks         = []
    sector_counts = {}
    skipped_fund  = 0
    skipped_sec   = 0
    skipped_earn  = 0
    skipped_corr  = 0
    skipped_port  = 0

    print(f"\nScanning {len(all_stocks)} stocks...")

    # Batch-download 2y price history for the ENTIRE (now S&P-500-sized) universe in
    # one pass (a handful of chunked/threaded calls) instead of one download per
    # stock inside the loop below — the main perf change needed once EXPAND_UNIVERSE
    # started pulling in ~450+ tickers instead of the original 139.
    print("Batch-downloading price history for the full universe...")
    stock_frames = batch_download(all_stocks, period="2y", interval="1d")
    print(f"  Got price data for {len(stock_frames)}/{len(all_stocks)} symbols.")

    for stock in all_stocks:
        if not passes_fundamental_filter(stock):
            skipped_fund += 1
            continue
        # Small pacing only around the remaining individual per-symbol calls below
        # (.info was just called above; .calendar is called by is_near_earnings).
        # Price data itself is already batched, so this no longer needs to be
        # 0.8s per stock like before.
        time.sleep(0.15)

        etf = active_sector_map.get(stock, "OTHER")
        if etf not in {"OTHER","GLD","SLV"}:
            if etf not in sector_strength_cache:
                sector_strength_cache[etf] = sector_is_strong(etf)
            if not sector_strength_cache[etf]:
                skipped_sec += 1
                continue

        if is_near_earnings(stock):
            skipped_earn += 1
            continue

        result = check_stock(stock, stock_frames.get(stock), spy_df, hot_sectors, sector_perf, active_sector_map, risk_pct=effective_risk)

        if result:
            # Portfolio concentration check
            blocked, block_reason = pick_blocked_by_portfolio(result, positions, sector_pct)
            if blocked:
                print(f"  {stock}: portfolio block — {block_reason}")
                skipped_port += 1
                continue

            sec = result['Sector']
            if sector_counts.get(sec, 0) >= MAX_PER_SECTOR:
                print(f"  {stock}: sector {sec} full ({MAX_PER_SECTOR}) — skip")
                skipped_corr += 1
                continue
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

            print(
                f"  ✅ {result['Symbol']:6} [{result['Sector']:5}]"
                f" score:{result['Score']:3}"
                f" setup:{result['Setup']:20}"
                f" MACD:{result['MACD']:12}"
                f" OBV:{result['OBV']:12}"
                f" ADX:{result['ADX']}"
            )
            picks.append(result)

    print(f"\n{'='*55}")
    print(f"Scanned:{len(all_stocks)} Fund❌:{skipped_fund} "
          f"Sector❌:{skipped_sec} Earn❌:{skipped_earn} "
          f"Corr❌:{skipped_corr} Port❌:{skipped_port} ✅:{len(picks)}")
    print(f"{'='*55}")

    if not picks:
        send_telegram(
            f"🔍 US Scan — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
            f"Market bullish | Breadth:{breadth_label} ({breadth}%)\n"
            f"No high-quality setups found — wait for next scan."
        )
        return

    # ---- Step 6: Separate quality picks from momentum-only alerts ----
    regular_picks = sorted(
        [p for p in picks if not p.get('MomentumAlert')],
        key=lambda x: (x['Score'], x['Reward$']), reverse=True
    )
    momentum_only = sorted(
        [p for p in picks if p.get('MomentumAlert')],
        key=lambda x: x['RVOL'], reverse=True
    )
    # Give Claude up to 10 candidates to reason over (more context = better ranking)
    candidates = regular_picks[:10]

    hot_str = ", ".join(sorted(hot_sectors)) if hot_sectors else "None"
    pos_str = f"{len(positions)} open" if positions else "None"

    # ---- Step 6b: Read recent signal performance from trade_log ----
    print("\nReading recent signal performance from trade log...")
    perf = get_recent_performance(60)
    if perf and perf.get("overall", {}).get("trades", 0) > 0:
        ov = perf["overall"]
        wr = ov.get("win_rate")
        print(f"  Last 60d: {int(wr*100)}% win rate over {ov['trades']} closed signals" if wr else "  No closed signals yet")

    # ---- Step 6c: Claude reasoning layer ----
    market_ctx = {
        "regime":        "Bullish (EMA50 + EMA200)",
        "vix_label":     vix_label,
        "breadth_label": breadth_label,
        "breadth":       breadth,
        "hot_str":       hot_str,
    }
    claude_output = {}
    if candidates:
        print(f"\nAsking Claude to reason over {len(candidates)} candidates...")
        claude_output = claude_reason(candidates, market_ctx, perf)
        if claude_output:
            print(f"  🧠 Market read: {claude_output.get('market_read','')[:80]}...")
            print(f"  🧠 Claude picks: {[p['symbol'] for p in claude_output.get('picks',[])]}")
            print(f"  🧠 Claude skip : {[s['symbol'] for s in claude_output.get('skip',[])]}")

    # ---- Step 6d: Re-rank top_picks by Claude's conviction order ----
    claude_picks_map = {}
    skip_set         = set()
    if claude_output and claude_output.get("picks"):
        claude_order     = [p["symbol"] for p in claude_output["picks"]]
        claude_picks_map = {p["symbol"]: p for p in claude_output["picks"]}
        skip_set         = {s["symbol"] for s in claude_output.get("skip", [])}
        # Reorder candidates to match Claude's ranking; append any Claude didn't mention
        ordered  = [c for sym in claude_order for c in candidates if c['Symbol'] == sym]
        fallback = [c for c in candidates if c['Symbol'] not in set(claude_order) | skip_set]
        top_picks = (ordered + fallback)[:TOP_PICKS]
    else:
        top_picks = candidates[:TOP_PICKS]

    # ---- Step 6e: News + 15-min on final top picks + momentum alerts ----
    sentiment_map = {}
    confirmed_map = {}
    all_checked   = top_picks + momentum_only[:3]
    print(f"\nRunning post-scan checks on {len(all_checked)} picks ({len(top_picks)} quality + {min(len(momentum_only),3)} momentum)...")
    for pick in all_checked:
        sym = pick['Symbol']
        if NEWS_SENTIMENT:
            ns, nl, nh = get_news_sentiment(sym)
            sentiment_map[sym] = (ns, nl, nh)
            print(f"  📰 {sym} news: {nl} (score {ns:+d})")
            time.sleep(0.3)
        # Skip 15-min for momentum alerts — they're already in motion
        if CHECK_15MIN and not pick.get('MomentumAlert'):
            ok, reason = passes_15min_check(sym, pick['Entry'])
            confirmed_map[sym] = (ok, reason)
            print(f"  {'✅' if ok else '❌'} {sym} 15m: {reason}")
            time.sleep(0.5)

    # ---- Step 7: Log picks ----
    log_picks(top_picks, confirmed_map, sentiment_map)

    # ---- Step 8: Telegram alerts ----

    # Market summary — include Claude's market read and confidence
    market_read  = claude_output.get("market_read", "") if claude_output else ""
    conf_label   = (
        f"{claude_output.get('overall_confidence','').upper()} — {claude_output.get('confidence_reason','')}"
        if claude_output else "N/A (Claude disabled)"
    )
    claude_summary = f"\n🧠 {market_read}\n🎯 Confidence: {conf_label}" if market_read else ""

    send_telegram(
        f"📊 US PRO SCAN — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
        f"{'='*34}\n"
        f"S&P    : Bullish (EMA50 + EMA200)\n"
        f"VIX    : {vix_label}\n"
        f"Breadth: {breadth_label} ({breadth}%)\n"
        f"Hot    : {hot_str}\n"
        f"Portfolio: {pos_str} | Heat: {total_heat:.0%}\n"
        f"Universe : {len(all_stocks)} stocks scanned\n"
        f"Quality  : {len(regular_picks)} setups | Momentum: {len(momentum_only)} alerts"
        f"{claude_summary}"
    )

    # --- Quality picks (Claude-ranked) ---
    for pick in top_picks:
        sym = pick['Symbol']
        rr  = round(pick['Reward$'] / pick['Risk$'], 1) if pick['Risk$'] > 0 else 0

        ns, nl, headlines = sentiment_map.get(sym, (0, "N/A", []))
        news_emoji = "📰✅" if ns >= 2 else "📰⚠️" if ns <= -2 else "📰"
        news_block = f"{news_emoji} News   : {nl}"
        if headlines:
            news_block += f"\n  → {headlines[0][:60]}"

        ok15, reason15 = confirmed_map.get(sym, (True, "Not checked"))
        conf_block = f"{'✅' if ok15 else '⚠️'} 15m    : {reason15}"

        # Claude conviction block
        c_pick = claude_picks_map.get(sym, {})
        if c_pick:
            icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(c_pick.get("conviction",""), "⚪")
            claude_block = f"\n🧠 {icon} {c_pick.get('conviction','').upper()}: {c_pick.get('reason','')}"
        else:
            claude_block = ""

        news_warn = "\n⚠️ NEGATIVE NEWS — review headlines before entering" if ns <= -2 else ""
        ext_warn  = f"\n⚠️ {pick['Extension']}" if pick['Extension'].startswith("⚠️") else ""

        msg = (
            f"{'='*34}\n"
            f"🚀 {sym}  [{pick['Sector']}]\n"
            f"Setup   : {pick['Setup']}\n"
            f"Score   : {pick['Score']}"
            f"{claude_block}\n"
            f"RVOL    : {pick['RVOL']}x | RSI: {pick['RSI']} | Move: {pick['Move3d']}\n"
            f"Candle  : {pick['Candle']}\n"
            f"MACD    : {pick['MACD']}\n"
            f"OBV     : {pick['OBV']}\n"
            f"ADX     : {pick['ADX']}\n"
            f"Squeeze : {pick['Squeeze']}\n"
            f"Extension: {pick['Extension']}\n"
            f"Sector Mom: 1D {pick['SectorDay']} | 1W {pick['SectorWeek']}\n"
            f"{news_block}\n"
            f"{conf_block}\n"
            f"Entry   : ${pick['Entry']}\n"
            f"Stop    : ${pick['Stop']}\n"
            f"Target  : ${pick['Target']}\n"
            f"Size    : {pick['Size']} shares\n"
            f"Invested: ${int(pick['Invested']):,} ({pick['AcctPct']}%)\n"
            f"Risk    : ${int(pick['Risk$']):,}\n"
            f"Reward  : ${int(pick['Reward$']):,}\n"
            f"RR      : 1:{rr}"
            f"{news_warn}"
            f"{ext_warn}\n"
            f"{'='*34}"
        )
        send_telegram(msg)
        time.sleep(0.5)

    # --- Claude skip list (if any) ---
    skip_list = claude_output.get("skip", []) if claude_output else []
    if skip_list:
        skip_lines = "\n".join(f"  • {s['symbol']}: {s['reason']}" for s in skip_list)
        send_telegram(f"⚠️ Claude flagged — lower priority:\n{skip_lines}")
        time.sleep(0.3)

    # --- Momentum-only alerts (clearly labeled higher-risk) ---
    for pick in momentum_only[:3]:
        sym = pick['Symbol']
        rr  = round(pick['Reward$'] / pick['Risk$'], 1) if pick['Risk$'] > 0 else 0

        ns, nl, headlines = sentiment_map.get(sym, (0, "N/A", []))
        news_emoji = "📰✅" if ns >= 2 else "📰⚠️" if ns <= -2 else "📰"
        news_block = f"{news_emoji} News: {nl}"
        if headlines:
            news_block += f"\n  → {headlines[0][:60]}"
        news_warn = "\n⚠️ NEGATIVE NEWS — review before entering" if ns <= -2 else ""

        msg = (
            f"{'='*34}\n"
            f"⚡ MOMENTUM ALERT — {sym}  [{pick['Sector']}]\n"
            f"⚠️ Below quality threshold — HIGHER RISK / SMALLER SIZE\n"
            f"{'='*34}\n"
            f"Setup    : {pick['Setup']}\n"
            f"Score    : {pick['Score']} (quality bar: {SCORE_THRESHOLD})\n"
            f"RVOL     : {pick['RVOL']}x | RSI: {pick['RSI']} | Move: {pick['Move3d']}\n"
            f"Extension: {pick['Extension']}\n"
            f"Sector Mom: 1D {pick['SectorDay']} | 1W {pick['SectorWeek']}\n"
            f"{news_block}\n"
            f"Entry   : ${pick['Entry']}\n"
            f"Stop    : ${pick['Stop']}\n"
            f"Target  : ${pick['Target']}\n"
            f"Size    : {pick['Size']} shares (consider ½ size)\n"
            f"Risk    : ${int(pick['Risk$']):,}\n"
            f"RR      : 1:{rr}"
            f"{news_warn}\n"
            f"{'='*34}"
        )
        send_telegram(msg)
        time.sleep(0.5)

# =========================================
# RUN
# =========================================

def is_market_hours():
    now_utc  = datetime.utcnow()
    if now_utc.weekday() > 4:
        return False
    time_val = now_utc.hour * 60 + now_utc.minute
    return (12 * 60) <= time_val <= (21 * 60 + 30)

if __name__ == "__main__":
    # NOTE: this used to self-loop every 30 min inside a single process until market
    # close. Combined with the workflow's 5 separate cron triggers spread across the
    # day, the FIRST trigger's loop already covered the entire trading day by itself
    # -- so each subsequent cron trigger started a redundant, fully-overlapping second
    # (third, fourth...) copy of the same day-long loop. That's very likely the source
    # of duplicate Telegram alerts, ~5x the intended Yahoo Finance API load, and the
    # trade_log.csv git-push races seen in practice (multiple overlapping processes
    # committing around the same time). Scheduling now lives ENTIRELY in the 5 cron
    # entries in .github/workflows/run_agent.yml -- this script does exactly one
    # scan-and-exit per invocation.
    print("🚀 US Professional Swing Trading Agent")
    print(f"Started at {datetime.utcnow().strftime('%H:%M UTC')}")

    run_agent()   # TEMP: bypassing market hours for Claude integration test
    # if is_market_hours():
    #     run_agent()
    # else:
    #     print("Outside market hours — skipping this run.")

    print("✅ Done.")
