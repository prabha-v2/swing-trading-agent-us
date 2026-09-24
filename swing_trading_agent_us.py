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
MAX_POSITION_PCT  = 0.15          # max 15% of account per trade
TOP_PICKS         = 5
MAX_PER_SECTOR    = 2

# Momentum Dip strategy (see check_stock / backtest.py)
MOM_TOP_PCT       = 0.30          # only stocks in the top 30% of the universe by 6-month momentum
DIP_RSI2_MAX      = 10            # 2-day RSI below this = short-term dip
DIP_STOP_ATR      = 2.5           # stop = entry - 2.5 x ATR(14)
DIP_MAX_HOLD      = 10            # exit after this many trading days if the 5-day-SMA exit hasn't fired

# Feature flags
EXPAND_UNIVERSE   = True          # Fetch S&P 500 dynamically (adds ~350 extra stocks)
NEWS_SENTIMENT    = True          # Score news headlines per pick
PORTFOLIO_FILE    = "positions.csv"
TRADE_LOG_FILE    = "trade_log.csv"
MAX_HOLD_DAYS     = 30            # close a signal as EXPIRED if neither stop nor target hit within this many calendar days
MIN_PERF_SAMPLE   = 8             # min unique closed signals before a setup/sector win rate is shown to Claude

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
    below). Sector data is context for alerts/Claude only — it doesn't filter picks.

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

def open_dip_symbols():
    """Symbols with a Momentum Dip signal still open in trade_log.csv (already alerted)."""
    log_file = Path(TRADE_LOG_FILE)
    if not log_file.exists():
        return set()
    try:
        with open(log_file, newline='') as f:
            return {r.get('symbol', '') for r in csv.DictReader(f)
                    if not r.get('outcome', '').strip() and r.get('setup') == 'Momentum Dip'}
    except Exception:
        return set()

def log_picks(picks, sentiment_map):
    """Append today's picks to trade_log.csv.

    Skips a symbol that already has an open Momentum Dip signal — re-logging it
    every run made one stop-out count as many losses in the win-rate stats.
    """
    log_file   = Path(TRADE_LOG_FILE)
    today_str  = datetime.now().strftime('%Y-%m-%d')
    file_exists = log_file.exists()

    # Load existing entries to avoid duplicates
    existing  = set()
    open_syms = open_dip_symbols()
    if file_exists:
        try:
            with open(log_file, newline='') as f:
                for row in csv.DictReader(f):
                    existing.add((row.get('date',''), row.get('symbol','')))
        except Exception:
            pass

    logged = 0

    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()

        for pick in picks:
            sym = pick['Symbol']
            if (today_str, sym) in existing or sym in open_syms:
                continue  # already logged today, or an earlier signal is still open
            rr = round(pick['Reward$'] / pick['Risk$'], 2) if pick['Risk$'] > 0 else 0
            writer.writerow({
                'date':           today_str,
                'symbol':         sym,
                'sector':         pick['Sector'],
                'setup':          pick['Setup'],
                'score':          pick['MomPct'],   # 6-month momentum percentile (0-100)
                'entry':          pick['Entry'],
                'stop':           pick['Stop'],
                'target':         pick['Target'],
                'size':           pick['Size'],
                'invested':       int(pick['Invested']),
                'risk_usd':       int(pick['Risk$']),
                'reward_usd':     int(pick['Reward$']),
                'rr':             rr,
                'news_sentiment': sentiment_map.get(sym, ('', 'N/A', []))[1],
                'confirmed_15m':  'N/A',
                'outcome':        '',
                'outcome_date':   '',
                'exit_price':     '',
                'pnl_usd':        '',
                'pnl_pct':        '',
            })
            logged += 1

    print(f"📋 Logged {logged} new picks to {TRADE_LOG_FILE} ({len(picks) - logged} already open/logged)")

def update_trade_outcomes():
    """
    For every open trade in trade_log.csv (outcome == ''), walk the daily bars
    after the signal date and record the exit. Fills in outcome, exit_price,
    pnl_usd, pnl_pct, outcome_date, and sends a Telegram SELL alert for
    Momentum Dip exits so the user knows to close the position.

    Momentum Dip: STOPPED if the low hits the stop; otherwise EXITED at the first
    close above the 5-day SMA, or at the close of trading day DIP_MAX_HOLD.
    Older setups: first stop or target hit; EXPIRED at the latest close after
    MAX_HOLD_DAYS. (Runs intraday, so today's bar is partial — an exit on it
    uses the current price.)
    """
    log_file = Path(TRADE_LOG_FILE)
    if not log_file.exists():
        return

    rows     = []
    updated  = 0
    today    = datetime.now()

    try:
        with open(log_file, newline='') as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"⚠️ Could not read trade log: {e}")
        return

    held_alerts  = []   # exits for stocks the user holds (listed in positions.csv)
    other_alerts = []   # exits for alerted signals the user didn't take
    held         = get_portfolio_positions()

    # One download per symbol, covering its oldest open signal
    open_rows  = [r for r in rows if not r.get('outcome', '').strip() and r.get('symbol')]
    earliest   = {}
    for r in open_rows:
        d = r.get('date', '')
        if d and (r['symbol'] not in earliest or d < earliest[r['symbol']]):
            earliest[r['symbol']] = d
    bars = {}
    for sym, d in earliest.items():
        try:
            # extra history so the 5-day SMA is defined from the first bar after the signal
            start = (pd.Timestamp(d) - timedelta(days=15)).strftime('%Y-%m-%d')
            df = yf.download(sym, start=start, interval="1d", progress=False)
            if df is None or df.empty:
                continue
            df = df.dropna()
            df.columns = df.columns.get_level_values(0)
            bars[sym] = df
        except Exception as e:
            print(f"  ⚠️ {sym} outcome check failed: {e}")

    for row in open_rows:
        sym    = row['symbol']
        entry  = float(row.get('entry', 0) or 0)
        stop   = float(row.get('stop', 0) or 0)
        target = float(row.get('target', 0) or 0)
        size   = int(float(row.get('size', 0) or 0))
        df     = bars.get(sym)

        if entry <= 0 or df is None:
            continue

        try:
            sig_date = pd.Timestamp(row['date'])
            # Bars after the signal day only — the signal-day bar includes prices from before the alert
            after = df[df.index.normalize() > sig_date]

            outcome    = ''
            exit_price = None
            exit_date  = None
            is_dip     = row.get('setup') == 'Momentum Dip'
            sma5       = df['Close'].rolling(5).mean()
            for n_bar, (ts, bar) in enumerate(after.iterrows(), 1):
                if is_dip:
                    o, lo, cl = float(bar['Open']), float(bar['Low']), float(bar['Close'])
                    if lo <= stop:
                        outcome, exit_price = 'STOPPED', min(o, stop)
                    elif cl > float(sma5.loc[ts]) or n_bar >= DIP_MAX_HOLD:
                        outcome, exit_price = 'EXITED', cl
                    if outcome:
                        exit_date = ts
                        break
                    continue
                o, hi, lo = float(bar['Open']), float(bar['High']), float(bar['Low'])
                # Stop checked first (conservative when both are inside one bar); gaps fill at the open
                if lo <= stop:
                    outcome, exit_price = 'STOPPED', min(o, stop)
                elif hi >= target:
                    outcome, exit_price = 'TARGET HIT', max(o, target)
                if outcome:
                    exit_date = ts
                    break

            close = float(df['Close'].iloc[-1])
            if not outcome and not is_dip and (today - sig_date).days >= MAX_HOLD_DAYS and not after.empty:
                outcome, exit_price, exit_date = 'EXPIRED', close, after.index[-1]

            if outcome:
                exit_price = round(exit_price, 2)
                pnl_usd = round((exit_price - entry) * size, 2)
                pnl_pct = round((exit_price - entry) / entry * 100, 2)
                row['outcome']      = outcome
                row['outcome_date'] = exit_date.strftime('%Y-%m-%d')
                row['exit_price']   = exit_price
                row['pnl_usd']      = pnl_usd
                row['pnl_pct']      = pnl_pct
                updated += 1
                emoji = {"TARGET HIT": "✅", "STOPPED": "❌"}.get(outcome, "✅" if pnl_usd > 0 else "⌛")
                print(f"  {emoji} {sym}: {outcome} | P&L ${pnl_usd:+.0f} ({pnl_pct:+.1f}%)")
                if is_dip:
                    if outcome == 'STOPPED':
                        why = "stop hit"
                    elif exit_price > float(sma5.loc[exit_date]):
                        why = "closed above 5-day SMA"
                    else:
                        why = f"{DIP_MAX_HOLD}-day time exit"
                    pos = held.get(sym.upper())
                    if pos:
                        my_pnl = (exit_price - pos['entry']) * pos['shares']
                        my_pct = (exit_price / pos['entry'] - 1) * 100 if pos['entry'] > 0 else 0
                        held_alerts.append(
                            f"📌 {sym} — SELL NOW ({why})\n"
                            f"  You hold {pos['shares']} @ ${pos['entry']:.2f} → ${exit_price:.2f} | "
                            f"${my_pnl:+,.0f} ({my_pct:+.1f}%)\n"
                            f"  After selling, remove {sym} from positions.csv"
                        )
                    else:
                        other_alerts.append(
                            f"{emoji} {sym} ({why}) — alert {row['date']} ${entry:.2f} → ${exit_price:.2f} | {pnl_pct:+.1f}%"
                        )
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

    if held_alerts:
        send_telegram("🔔 SELL — YOUR POSITIONS:\n" + "\n".join(held_alerts))
    if other_alerts:
        send_telegram(
            "ℹ️ Momentum Dip signals closed (not in positions.csv — ignore unless you hold them):\n"
            + "\n".join(other_alerts)
        )

def print_trade_stats():
    """Print win-rate / P&L for closed Momentum Dip signals (a win = closed at a profit)."""
    log_file = Path(TRADE_LOG_FILE)
    if not log_file.exists():
        return

    try:
        with open(log_file, newline='') as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return

    closed = [r for r in rows
              if r.get('setup') == 'Momentum Dip' and r.get('outcome', '').strip() and r.get('pnl_pct')]
    if not closed:
        print("\n📈 Momentum Dip history: no closed signals yet")
        return

    pnl_pct = [float(r['pnl_pct']) for r in closed]
    pnl_usd = [float(r['pnl_usd']) for r in closed if r.get('pnl_usd')]
    win_r   = sum(p > 0 for p in pnl_pct) / len(pnl_pct) * 100
    print(
        f"\n📈 Momentum Dip history: {len(closed)} closed | Win rate: {win_r:.0f}% | "
        f"Avg: {sum(pnl_pct) / len(pnl_pct):+.2f}%/trade | Net P&L: ${sum(pnl_usd):+,.0f}"
    )

# =========================================
# MAIN TECHNICAL SCANNER
# =========================================

def momentum_6m(df):
    """6-month momentum skipping the latest month: close 21 bars ago vs 126 bars ago."""
    try:
        if df is None or len(df) < 130:
            return None
        c = df['Close'].dropna()
        if len(c) < 130:
            return None
        return float(c.iloc[-22] / c.iloc[-127] - 1)
    except Exception:
        return None

def check_stock(symbol, df, mom_pct, sector_perf, active_sector_map, risk_pct=RISK_PER_TRADE):
    """
    Momentum Dip setup: a short, sharp pullback in one of the market's strongest stocks.

      - Long-term uptrend : close > 200-day SMA
      - Leader            : 6-month momentum in the top MOM_TOP_PCT of the scanned universe
                            (`mom_pct` = this stock's percentile rank, 0-1, computed in run_agent)
      - Short-term dip    : 2-day RSI < DIP_RSI2_MAX
      - Stop              : entry - DIP_STOP_ATR x ATR(14)
      - Exit              : first close above the 5-day SMA, or after DIP_MAX_HOLD trading days

    Backtested 2018-2026 on the S&P 500 + curated universe (see backtest.py): ~+0.5% per
    trade over ~3.4 days, ~68% winners, positive in 8 of 9 years. The old multi-indicator
    score it replaced did no better than picking random stocks.
    """
    try:
        if df is None or df.empty:
            return None
        df = df.dropna()
        if len(df) < 210 or mom_pct is None or mom_pct < 1 - MOM_TOP_PCT:
            return None

        close = df['Close']
        price = float(close.iloc[-1])
        if price < 5.0:
            return None
        avg_dv = float((close * df['Volume']).iloc[-20:].mean())
        if avg_dv < 2_000_000:
            return None

        sma200 = float(close.rolling(200).mean().iloc[-1])
        if price <= sma200:
            return None

        rsi2 = float(ta.momentum.rsi(close, window=2).iloc[-1])
        if not rsi2 < DIP_RSI2_MAX:
            return None

        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - close.shift()).abs(),
            (df['Low'] - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr  = float(tr.rolling(14).mean().iloc[-1])
        sma5 = float(close.rolling(5).mean().iloc[-1])

        entry = price
        stop  = entry - DIP_STOP_ATR * atr
        risk  = entry - stop
        if risk <= 0 or risk > entry * 0.15:
            return None

        max_by_dollars = int((ACCOUNT_SIZE * MAX_POSITION_PCT) / entry)
        size           = min(int((ACCOUNT_SIZE * risk_pct) / risk), max_by_dollars)
        if size <= 0:
            return None
        invested = round(entry * size, 0)

        stock_sector = active_sector_map.get(symbol, "OTHER")
        sp = sector_perf.get(stock_sector, {}) if sector_perf else {}

        return {
            "Symbol":     symbol,
            "Sector":     stock_sector,
            "Setup":      "Momentum Dip",
            "Mom6m":      round(momentum_6m(df) * 100, 1),
            "MomPct":     int(round(mom_pct * 100)),
            "RSI2":       round(rsi2, 1),
            "RSI":        round(float(ta.momentum.rsi(close, window=14).iloc[-1]), 1),
            "Move3d":     f"{price / float(close.iloc[-4]) - 1:+.1%}",
            "AboveSMA200": f"{price / sma200 - 1:+.1%}",
            "ATRpct":     round(atr / price * 100, 1),
            "SectorDay":  f"{sp['ret_1d']:+.1%}" if 'ret_1d' in sp else "n/a",
            "SectorWeek": f"{sp['ret_1w']:+.1%}" if 'ret_1w' in sp else "n/a",
            "Entry":      round(entry, 2),
            "Stop":       round(stop, 2),
            "Target":     round(sma5, 2),       # exit reference: first close above the 5-day SMA
            "Size":       size,
            "Invested":   invested,
            "AcctPct":    round((invested / ACCOUNT_SIZE) * 100, 1),
            "Risk$":      round(risk * size, 0),
            "Reward$":    round(max(sma5 - entry, 0) * size, 0),
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
    "OKTA":"XLK",  "S":"XLK",     "SNOW":"XLK",
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
    Only Momentum Dip signals count (older setups were retired); a win is a
    signal that closed at a profit.
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
        recent  = recent[recent["setup"] == "Momentum Dip"]
        closed  = recent[recent["outcome"].notna() & recent["pnl_pct"].notna()].copy()

        def stats(rows):
            total = len(rows)
            wins  = int((rows["pnl_pct"] > 0).sum())
            return {
                "trades":   total,
                "wins":     wins,
                "win_rate": round(wins / total, 2) if total > 0 else None,
                "avg_pct":  round(float(rows["pnl_pct"].mean()), 2) if total > 0 else None,
            }

        by_sector = {g: stats(gdf) for g, gdf in closed.groupby("sector") if len(gdf) >= MIN_PERF_SAMPLE}

        last5  = closed.sort_values("outcome_date").tail(5)
        streak = " → ".join(
            "✅W" if p > 0 else "❌L"
            for p in last5["pnl_pct"]
        ) if not last5.empty else "No closed trades yet"

        return {
            "window_days":    n_days,
            "total_signals":  len(recent),
            "overall":        stats(closed),
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
            f"Momentum Dip live results (last {perf.get('window_days', 60)}d): "
            + (f"{int(wr*100)}% winners, avg {overall.get('avg_pct', 0):+.2f}%/trade across {n} closed signals"
               if wr is not None else "Not enough data yet")
        )
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
            f"{i}. {c['Symbol']} [{c['Sector']}] | 6m momentum:{c['Mom6m']:+.0f}% "
            f"(top {100 - c['MomPct']}%) | RSI2:{c['RSI2']} RSI14:{c['RSI']} 3d move:{c['Move3d']} "
            f"| vs SMA200:{c['AboveSMA200']} ATR:{c['ATRpct']}% "
            f"| Entry:${c['Entry']} Stop:${c['Stop']} 5d-SMA exit ref:${c['Target']} "
            f"| Sector 1D:{c['SectorDay']} 1W:{c['SectorWeek']}"
        )

    prompt = f"""You are a professional swing trading analyst reviewing today's scan for US stocks.
The user will review these alerts and decide which trades to actually take — they do NOT enter every signal.

STRATEGY — "Momentum Dip": buy a short, sharp pullback (2-day RSI < {DIP_RSI2_MAX}) in stocks that are
above their 200-day SMA and in the top {int(MOM_TOP_PCT*100)}% of the universe by 6-month momentum. Exit at the
first close above the 5-day SMA (typically 2-5 days), stop at {DIP_STOP_ATR}x ATR, max {DIP_MAX_HOLD} days.
Backtest 2018-2026: ~+0.5%/trade, ~68% winners, positive in 8 of 9 years. The dip IS the setup —
weak short-term indicators (low RSI, red candles, bearish MACD) are expected and are not a reason to pass.
Candidates are already ordered by 6-month momentum, which is the ranking that was backtested.

Your job: add judgement the price data can't see — e.g. a dip caused by a fundamental break
(earnings/guidance collapse, fraud, downgrade on a broken thesis) rather than ordinary profit-taking,
several picks that are really one bet, or unusual market stress.

MARKET CONDITIONS:
- S&P 500  : {market_ctx.get('regime', 'N/A')}
- VIX      : {market_ctx.get('vix_label', 'N/A')}
- Breadth  : {market_ctx.get('breadth_label', 'N/A')} ({market_ctx.get('breadth', 'N/A')}% above EMA50)
- Hot sectors: {market_ctx.get('hot_str', 'None')}

RECENT SIGNAL PERFORMANCE (screener accuracy, not user P&L; sectors with fewer than
{MIN_PERF_SAMPLE} closed signals are omitted as too small to judge):
{chr(10).join(perf_lines) if perf_lines else 'No performance data yet — first scan.'}

TODAY'S CANDIDATES ({len(candidates)} stocks, ranked by 6-month momentum):
{chr(10).join(cand_lines)}

Respond ONLY with a valid JSON object — no markdown fences, no extra text:
{{
  "market_read": "<2 sentences: current market tone and what it means for swing trades today>",
  "picks": [
    {{"symbol": "TICKER", "conviction": "high|medium|low",
      "reason": "<1-2 sentences: why this dip looks like a buyable pullback, or what to watch>"}}
  ],
  "cautions": [
    {{"symbol": "TICKER", "reason": "<1 sentence: the specific concern>"}}
  ],
  "overall_confidence": "high|medium|low",
  "confidence_reason": "<one sentence>"
}}

Rules:
- picks: give a conviction note for EVERY candidate, in the order given.
- cautions: only for a concrete, specific concern (not "RSI is low" or "MACD bearish" — that is the setup).
  Cautions are shown to the user as warnings; they do not remove the pick. Empty list is fine.
- Be specific — cite actual numbers. Don't invent news; if you don't know why a stock dipped, say so.
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

    # ---- Step 1: Market context ----
    # Informational only: the S&P-trend and breadth gates made Momentum Dip results
    # worse in backtests (dips in leaders during market pullbacks are among the best trades).
    spx_bullish  = market_is_bullish()
    regime_label = "Bullish (above EMA50 + EMA200)" if spx_bullish else "Weak (below EMA50/EMA200)"

    vix = get_vix()
    effective_risk = RISK_PER_TRADE * (0.5 if vix > 25 else 1.0)
    vix_label      = f"Elevated ({vix:.0f}) — half size" if vix > 25 else f"Normal ({vix:.0f})"

    print("\nChecking market breadth...")
    breadth       = get_market_breadth()
    breadth_label = "Strong" if breadth >= 60 else "Mixed" if breadth >= 40 else "Weak"

    print("\nChecking sector rotation...")
    hot_sectors, sector_perf, _ = get_sector_rotation()

    # ---- Step 2: Build universe ----
    active_sector_map = get_dynamic_universe(sector_map) if EXPAND_UNIVERSE else dict(sector_map)

    # Inject today's top movers so stocks outside the S&P 500 or curated list
    # still get scanned (e.g. mid-caps, recent IPOs).
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

    # ---- Step 4: Price history + 6-month momentum rank across the universe ----
    print(f"\nScanning {len(all_stocks)} stocks...")
    print("Batch-downloading price history for the full universe...")
    stock_frames = batch_download(all_stocks, period="2y", interval="1d")
    print(f"  Got price data for {len(stock_frames)}/{len(all_stocks)} symbols.")

    mom = {}
    for sym, df in stock_frames.items():
        m = momentum_6m(df)
        if m is None:
            continue
        c = df['Close'].dropna()
        # rank only among tradable names, matching the backtest universe
        if float(c.iloc[-1]) >= 5.0 and float((c * df['Volume']).iloc[-20:].mean()) >= 2_000_000:
            mom[sym] = m
    mom_pct = pd.Series(mom).rank(pct=True).to_dict() if mom else {}

    # ---- Step 5: Technical scan (cheap), then per-stock lookups only on hits ----
    raw_hits = []
    for stock in all_stocks:
        result = check_stock(stock, stock_frames.get(stock), mom_pct.get(stock),
                             sector_perf, active_sector_map, risk_pct=effective_risk)
        if result:
            raw_hits.append(result)
    # Strongest momentum first — the ranking that was backtested
    raw_hits.sort(key=lambda x: x['Mom6m'], reverse=True)
    print(f"  {len(raw_hits)} Momentum Dip setups before earnings/fundamental/portfolio checks")

    already_open  = open_dip_symbols()   # alerted in an earlier run and not exited yet
    picks         = []
    sector_counts = {}
    skipped_open  = 0
    skipped_fund  = 0
    skipped_earn  = 0
    skipped_corr  = 0
    skipped_port  = 0

    for result in raw_hits:
        stock = result['Symbol']
        sec   = result['Sector']
        if stock in already_open:
            skipped_open += 1
            continue
        if sector_counts.get(sec, 0) >= MAX_PER_SECTOR:
            skipped_corr += 1
            continue
        if not passes_fundamental_filter(stock):
            skipped_fund += 1
            continue
        if is_near_earnings(stock):
            skipped_earn += 1
            continue
        blocked, block_reason = pick_blocked_by_portfolio(result, positions, sector_pct)
        if blocked:
            print(f"  {stock}: portfolio block — {block_reason}")
            skipped_port += 1
            continue
        time.sleep(0.15)   # pacing for the per-symbol .info/.calendar calls above

        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        print(
            f"  ✅ {stock:6} [{sec:5}] 6m mom:{result['Mom6m']:+6.1f}% (top {100 - result['MomPct']}%)"
            f" RSI2:{result['RSI2']:5.1f} 3d:{result['Move3d']}"
        )
        picks.append(result)
        if len(picks) >= TOP_PICKS:
            break

    print(f"\n{'='*55}")
    print(f"Scanned:{len(all_stocks)} Setups:{len(raw_hits)} AlreadyOpen:{skipped_open} Fund❌:{skipped_fund} "
          f"Earn❌:{skipped_earn} Corr❌:{skipped_corr} Port❌:{skipped_port} ✅:{len(picks)}")
    print(f"{'='*55}")

    if not picks:
        send_telegram(
            f"🔍 US Scan — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
            f"S&P: {regime_label} | Breadth: {breadth_label} ({breadth}%)\n"
            f"No new Momentum Dip setups ({skipped_open} already alerted and still open). Wait for next scan."
        )
        return

    top_picks = picks
    hot_str = ", ".join(sorted(hot_sectors)) if hot_sectors else "None"
    pos_str = f"{len(positions)} open" if positions else "None"

    # ---- Step 6: Read recent signal performance from trade_log ----
    print("\nReading recent signal performance from trade log...")
    perf = get_recent_performance(60)
    if perf and perf.get("overall", {}).get("trades", 0) > 0:
        ov = perf["overall"]
        print(f"  Last 60d: {int(ov['win_rate']*100)}% winners, avg {ov['avg_pct']:+.2f}% over {ov['trades']} closed signals")

    # ---- Step 6b: Claude commentary (annotates picks; does not remove or reorder them) ----
    market_ctx = {
        "regime":        regime_label,
        "vix_label":     vix_label,
        "breadth_label": breadth_label,
        "breadth":       breadth,
        "hot_str":       hot_str,
    }
    print(f"\nAsking Claude to review {len(top_picks)} picks...")
    claude_output = claude_reason(top_picks, market_ctx, perf)
    if claude_output:
        print(f"  🧠 Market read: {claude_output.get('market_read','')[:80]}...")
        print(f"  🧠 Cautions: {[c['symbol'] for c in claude_output.get('cautions',[])]}")
    claude_picks_map = {p["symbol"]: p for p in claude_output.get("picks", [])} if claude_output else {}
    caution_map      = {c["symbol"]: c["reason"] for c in claude_output.get("cautions", [])} if claude_output else {}

    # ---- Step 6c: News on final picks ----
    sentiment_map = {}
    if NEWS_SENTIMENT:
        print(f"\nChecking news on {len(top_picks)} picks...")
        for pick in top_picks:
            sym = pick['Symbol']
            ns, nl, nh = get_news_sentiment(sym)
            sentiment_map[sym] = (ns, nl, nh)
            print(f"  📰 {sym} news: {nl} (score {ns:+d})")
            time.sleep(0.3)

    # ---- Step 7: Log picks ----
    log_picks(top_picks, sentiment_map)

    # ---- Step 8: Telegram alerts ----
    market_read  = claude_output.get("market_read", "") if claude_output else ""
    conf_label   = (
        f"{claude_output.get('overall_confidence','').upper()} — {claude_output.get('confidence_reason','')}"
        if claude_output else "N/A (Claude disabled)"
    )
    claude_summary = f"\n🧠 {market_read}\n🎯 Confidence: {conf_label}" if market_read else ""

    send_telegram(
        f"📊 US MOMENTUM DIP SCAN — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
        f"{'='*34}\n"
        f"S&P    : {regime_label}\n"
        f"VIX    : {vix_label}\n"
        f"Breadth: {breadth_label} ({breadth}%)\n"
        f"Hot    : {hot_str}\n"
        f"Portfolio: {pos_str} | Heat: {total_heat:.0%}\n"
        f"Universe : {len(all_stocks)} stocks | Setups: {len(raw_hits)} | Already open: {skipped_open} | New: {len(top_picks)}\n"
        f"Rules  : buy near close · stop {DIP_STOP_ATR}×ATR · sell on first close above 5-day SMA "
        f"(max {DIP_MAX_HOLD} days) — the bot sends SELL alerts"
        f"{claude_summary}"
    )

    for pick in top_picks:
        sym = pick['Symbol']

        ns, nl, headlines = sentiment_map.get(sym, (0, "N/A", []))
        news_emoji = "📰✅" if ns >= 2 else "📰⚠️" if ns <= -2 else "📰"
        news_block = f"{news_emoji} News   : {nl}"
        if headlines:
            news_block += f"\n  → {headlines[0][:60]}"

        c_pick = claude_picks_map.get(sym, {})
        if c_pick:
            icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(c_pick.get("conviction",""), "⚪")
            claude_block = f"\n🧠 {icon} {c_pick.get('conviction','').upper()}: {c_pick.get('reason','')}"
        else:
            claude_block = ""
        caution_block = f"\n⚠️ Claude caution: {caution_map[sym]}" if sym in caution_map else ""
        news_warn     = "\n⚠️ NEGATIVE NEWS — check why it dipped before entering" if ns <= -2 else ""

        msg = (
            f"{'='*34}\n"
            f"🎯 {sym}  [{pick['Sector']}] — Momentum Dip\n"
            f"6m Mom  : {pick['Mom6m']:+.1f}% (top {100 - pick['MomPct']}% of universe)\n"
            f"Dip     : RSI2 {pick['RSI2']} | 3-day {pick['Move3d']} | RSI14 {pick['RSI']}\n"
            f"Trend   : {pick['AboveSMA200']} above 200-day SMA"
            f"{claude_block}\n"
            f"Sector Mom: 1D {pick['SectorDay']} | 1W {pick['SectorWeek']}\n"
            f"{news_block}\n"
            f"Entry   : ${pick['Entry']}\n"
            f"Stop    : ${pick['Stop']} ({DIP_STOP_ATR}×ATR, ATR {pick['ATRpct']}%)\n"
            f"Exit    : first close above 5-day SMA (now ${pick['Target']}) or day {DIP_MAX_HOLD}\n"
            f"Size    : {pick['Size']} shares\n"
            f"Invested: ${int(pick['Invested']):,} ({pick['AcctPct']}%)\n"
            f"Risk    : ${int(pick['Risk$']):,}"
            f"{caution_block}"
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

    if is_market_hours():
        run_agent()
    else:
        print("Outside market hours — skipping this run.")

    print("✅ Done.")
