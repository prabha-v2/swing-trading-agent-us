import yfinance as yf
import pandas as pd
import ta
import time
import os
import requests
from datetime import datetime

# =========================================
# SETTINGS
# =========================================

ACCOUNT_SIZE    = 30000
RISK_PER_TRADE  = 0.01         # 1% risk = $300 per trade
RR_RATIO        = 2.5
MAX_ATR_STOP    = 3.0
MAX_POSITION    = 500
SCORE_THRESHOLD = 18           # Raised for professional quality
TOP_PICKS       = 5

TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID         = os.environ.get("CHAT_ID", "")

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
# PROFESSIONAL LAYER 1
# MARKET BREADTH CHECK
# Checks % of S&P 500 stocks above 50-day MA
# Professionals never trade when breadth is weak
# =========================================

def get_market_breadth():
    """
    Downloads a sample of 50 S&P 500 stocks
    and checks what % are above their 50-day MA.
    > 60% = healthy market (green light)
    40-60% = mixed (proceed with caution)
    < 40% = weak breadth (avoid new longs)
    """
    sp500_sample = [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","LLY","JPM","V",
        "XOM","UNH","MA","JNJ","PG","HD","MRK","ABBV","CVX","KO",
        "PEP","AVGO","COST","WMT","BAC","CRM","TMO","ORCL","ACN","MCD",
        "CSCO","ABT","LIN","DHR","NEE","TXN","PM","WFC","UNP","RTX",
        "BMY","AMGN","QCOM","HON","INTU","IBM","GE","CAT","SPGI","ELV"
    ]
    above_50 = 0
    total    = 0
    for sym in sp500_sample:
        try:
            df = yf.download(sym, period="6mo", interval="1d", progress=False)
            df = df.dropna()
            df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 50:
                continue
            df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50)
            if float(df['Close'].iloc[-1]) > float(df['EMA50'].iloc[-1]):
                above_50 += 1
            total += 1
            time.sleep(0.3)
        except Exception:
            continue

    if total == 0:
        return 50  # neutral if no data
    pct = round((above_50 / total) * 100, 1)
    print(f"Market Breadth: {above_50}/{total} stocks above EMA50 = {pct}%")
    return pct

# =========================================
# PROFESSIONAL LAYER 2
# SECTOR ROTATION DETECTOR
# Checks if money is flowing INTO the sector
# RIGHT NOW (not just over the past year)
# =========================================

def get_sector_rotation():
    """
    Checks 1-week and 1-month performance of
    all major sector ETFs to find which sectors
    are seeing active money rotation RIGHT NOW.
    Returns a set of currently hot sector ETFs.
    """
    sector_etfs = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLU": "Utilities",
        "XLB": "Materials",
        "XLRE":"Real Estate",
        "XLY": "Consumer Disc",
        "XLP": "Consumer Staples",
        "XLC": "Communication",
        "SMH": "Semiconductors",
        "ITA": "Defense",
        "TAN": "Solar",
        "URA": "Nuclear",
        "XBI": "Biotech",
    }
    hot_sectors = set()
    sector_perf = {}

    for etf, name in sector_etfs.items():
        try:
            df = yf.download(etf, period="3mo", interval="1d", progress=False)
            df = df.dropna()
            df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 20:
                continue
            # 1-week return (5 trading days)
            ret_1w = float(df['Close'].pct_change(5).iloc[-1])
            # 1-month return (21 trading days)
            ret_1m = float(df['Close'].pct_change(21).iloc[-1])
            # Volume trend (recent vs older)
            avg_vol_recent = float(df['Volume'].iloc[-5:].mean())
            avg_vol_older  = float(df['Volume'].iloc[-21:-5].mean())
            vol_trend      = avg_vol_recent / avg_vol_older if avg_vol_older > 0 else 1

            sector_perf[etf] = {
                "name":     name,
                "ret_1w":   ret_1w,
                "ret_1m":   ret_1m,
                "vol_trend": vol_trend
            }
            time.sleep(0.3)
        except Exception:
            continue

    if not sector_perf:
        return hot_sectors

    # Calculate median returns as benchmark
    all_1w = [v['ret_1w'] for v in sector_perf.values()]
    all_1m = [v['ret_1m'] for v in sector_perf.values()]
    med_1w = sorted(all_1w)[len(all_1w)//2]
    med_1m = sorted(all_1m)[len(all_1m)//2]

    print("\nSector Rotation (1W | 1M | Vol Trend):")
    for etf, data in sorted(sector_perf.items(),
                             key=lambda x: x[1]['ret_1w'], reverse=True):
        is_hot = (data['ret_1w'] > med_1w and
                  data['ret_1m'] > med_1m and
                  data['vol_trend'] > 0.9)
        if is_hot:
            hot_sectors.add(etf)
        flag = "🔥" if is_hot else "  "
        print(f"  {flag} {etf:6} {data['name']:20}"
              f" 1W:{data['ret_1w']:+.1%}"
              f" 1M:{data['ret_1m']:+.1%}"
              f" Vol:{data['vol_trend']:.2f}x")

    return hot_sectors

# =========================================
# PROFESSIONAL LAYER 3
# CANDLE QUALITY FILTER
# Checks the shape and pattern of last 3 candles
# Professionals only enter on quality candle setups
# =========================================

def candle_quality_score(df):
    """
    Analyses the last 3 candles for quality.
    Returns a score from -4 to +6.

    Bullish patterns:
    + Strong bullish close (close in top 30% of range)
    + Inside bar breakout (today > yesterday's high)
    + No upper wick rejection (wick < 30% of range)
    + Recent candle body > ATR * 0.5 (strong move)
    + Gap up from yesterday's close

    Bearish patterns (penalties):
    - Shooting star / doji (indecision)
    - Long upper wick (rejection)
    - Close below open for 2 of last 3 days
    """
    score = 0

    try:
        c1 = df.iloc[-1]   # today
        c2 = df.iloc[-2]   # yesterday
        c3 = df.iloc[-3]   # 2 days ago

        atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else \
              float(c1['High'] - c1['Low'])

        # --- Today's candle analysis ---
        today_range  = float(c1['High'] - c1['Low'])
        today_body   = abs(float(c1['Close'] - c1['Open']))
        today_upper  = float(c1['High']) - max(float(c1['Close']), float(c1['Open']))
        today_lower  = min(float(c1['Close']), float(c1['Open'])) - float(c1['Low'])
        close_pos    = ((float(c1['Close']) - float(c1['Low'])) /
                        today_range) if today_range > 0 else 0.5

        # Strong bullish close (in top 30% of day's range)
        if close_pos > 0.70:
            score += 2
        elif close_pos > 0.50:
            score += 1
        elif close_pos < 0.30:
            score -= 2  # weak close near lows

        # Long upper wick rejection (bearish)
        if today_range > 0 and today_upper / today_range > 0.40:
            score -= 1  # shooting star / bearish rejection

        # Strong body (momentum candle)
        if today_body > atr * 0.5:
            score += 1

        # Gap up from yesterday (institutional buying)
        if float(c1['Open']) > float(c2['Close']) * 1.005:
            score += 1

        # Inside bar breakout (today breaks above yesterday's high)
        if float(c1['Close']) > float(c2['High']):
            score += 1

        # --- Last 3 candles trend ---
        bullish_candles = sum([
            float(c1['Close']) > float(c1['Open']),
            float(c2['Close']) > float(c2['Open']),
            float(c3['Close']) > float(c3['Open']),
        ])
        if bullish_candles == 3:
            score += 1   # three consecutive bullish candles
        elif bullish_candles <= 1:
            score -= 1   # mostly bearish candles recently

        # Doji check (indecision — body < 10% of range)
        if today_range > 0 and today_body / today_range < 0.10:
            score -= 1   # doji = uncertainty, avoid entry

    except Exception as e:
        print(f"    Candle analysis error: {e}")

    return score

# =========================================
# PROFESSIONAL LAYER 4
# FUNDAMENTAL FILTER
# =========================================

SKIP_FUNDAMENTAL = {
    "SMH","SOXX","ITA","XAR","TAN","ICLN","URA","URNM",
    "ARKX","QTUM","XBI","XLU","XLI","GLD","IAU","SLV",
    "COPX","UFO","XLK","XLF","XLE","XLV","XLB","XLRE",
    "XLY","XLP","XLC"
}

def passes_fundamental_filter(symbol):
    if symbol in SKIP_FUNDAMENTAL:
        return True
    try:
        info = yf.Ticker(symbol).info
        if not info:
            return True

        # EPS — not deeply loss making
        eps = info.get("trailingEps", None)
        if eps is not None and eps < -5:
            print(f"    ❌ {symbol}: EPS {eps:.1f} — skip")
            return False

        # Debt/Equity < 300%
        de = info.get("debtToEquity", None)
        if de is not None and de > 300:
            print(f"    ❌ {symbol}: D/E {de:.0f} — skip")
            return False

        # Market cap > $1B
        mktcap = info.get("marketCap", None)
        if mktcap is not None and mktcap < 1_000_000_000:
            print(f"    ❌ {symbol}: mktcap too small — skip")
            return False

        # Revenue > 0
        rev = info.get("totalRevenue", None)
        if rev is not None and rev <= 0:
            print(f"    ❌ {symbol}: no revenue — skip")
            return False

        return True
    except Exception:
        return True

# =========================================
# MARKET TREND FILTER — S&P 500
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
    ema50  = float(latest['EMA50'])
    ema200 = float(latest['EMA200'])
    print(f"S&P: {close:.0f} | EMA50: {ema50:.0f} | EMA200: {ema200:.0f}")
    return close > ema50 and close > ema200

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

def is_near_earnings(symbol, days=5):
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
    except Exception:
        pass
    return False

# =========================================
# MAIN TECHNICAL SCANNER
# =========================================

def check_stock(symbol, spy_df, hot_sectors):
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df is None or df.empty or len(df) < 250:
            return None

        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 250:
            return None

        price = float(df['Close'].iloc[-1])
        if price < 5.0:
            return None

        # Liquidity — $2M+ daily
        avg_dv = float(
            df['Close'].iloc[-20:].mean() * df['Volume'].iloc[-20:].mean()
        )
        if avg_dv < 2_000_000:
            return None

        # Indicators
        df['EMA10']  = ta.trend.ema_indicator(df['Close'], window=10)
        df['EMA20']  = ta.trend.ema_indicator(df['Close'], window=20)
        df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
        df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
        df['RSI']    = ta.momentum.rsi(df['Close'], window=14)
        df['AvgVol'] = df['Volume'].rolling(20).mean()
        df['HH20']   = df['High'].rolling(20).max()
        df['High52']  = df['High'].rolling(252).max()
        df['TR'] = (
            df['High'] - df['Low']
        ).combine(abs(df['High'] - df['Close'].shift(1)), max
        ).combine(abs(df['Low']  - df['Close'].shift(1)), max)
        df['ATR'] = df['TR'].rolling(14).mean()

        # Relative strength vs SPY
        s3  = df['Close'].pct_change(63).iloc[-1]
        s6  = df['Close'].pct_change(126).iloc[-1]
        s12 = df['Close'].pct_change(252).iloc[-1]
        n3  = spy_df['Close'].pct_change(63).iloc[-1]
        n6  = spy_df['Close'].pct_change(126).iloc[-1]
        n12 = spy_df['Close'].pct_change(252).iloc[-1]
        rs  = sum([s3 > n3, s6 > n6, s12 > n12])

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        score = 0

        # ---- RELATIVE STRENGTH ----
        if rs >= 2:                                              score += 2
        if rs == 3:                                              score += 1

        # ---- EMA TREND STACK ----
        if latest['EMA10']  > latest['EMA20']:                  score += 2
        if latest['EMA20']  > latest['EMA50']:                  score += 2
        if latest['EMA50']  > latest['EMA200']:                 score += 2
        if latest['Close']  > latest['EMA50']:                  score += 1
        if latest['Close']  > latest['EMA200']:                 score += 2

        # ---- RSI ----
        rsi = float(latest['RSI'])
        if prev['RSI'] < 50 and rsi > 50:                       score += 2
        if 50 < rsi < 70:                                        score += 1
        if rsi > 75:                                             score -= 1

        # ---- BREAKOUT ----
        if latest['Close'] > prev['HH20']:                      score += 2
        ath_dist = latest['Close'] / latest['High52']
        if ath_dist > 0.90:                                      score += 2
        elif ath_dist > 0.80:                                    score += 1

        # ---- VOLUME ----
        if latest['AvgVol'] > 0:
            rvol = latest['Volume'] / latest['AvgVol']
            if rvol > 2.0:                                       score += 2
            elif rvol > 1.5:                                     score += 1

        # ---- ATR EXPANDING ----
        if latest['ATR'] > df['ATR'].iloc[-5]:                  score += 1

        # ---- ENTRY QUALITY ----
        dist = (latest['Close'] - latest['EMA20']) / latest['EMA20']
        if dist < 0.05:                                          score += 2
        elif dist < 0.08:                                        score += 1
        elif dist > 0.20:                                        score -= 1

        # ---- 60-DAY OUTPERFORMANCE ----
        if len(df) >= 60 and len(spy_df) >= 60:
            sr = float(df['Close'].squeeze().pct_change(60).iloc[-1])
            nr = float(spy_df['Close'].squeeze().pct_change(60).iloc[-1])
            if sr > nr * 1.5:                                    score += 2
            elif sr > nr:                                        score += 1

        # ---- DISTRIBUTION PENALTY ----
        recent_high = df['High'].iloc[-10:].max()
        if latest['Close'] < recent_high * 0.85:
            score -= 2

        # ---- PROFESSIONAL LAYER: CANDLE QUALITY ----
        candle_score = candle_quality_score(df)
        score += candle_score

        # ---- PROFESSIONAL LAYER: SECTOR ROTATION BONUS ----
        stock_sector = sector_map.get(symbol, "OTHER")
        if stock_sector in hot_sectors:
            score += 2   # money actively flowing into this sector NOW
            print(f"    🔥 {symbol} sector {stock_sector} is HOT — +2")

        if score < SCORE_THRESHOLD:
            return None

        # ---- POSITION SIZING ----
        entry        = float(latest['Close'])
        atr          = float(latest['ATR'])
        five_bar_low = float(df['Low'].iloc[-5:].min())
        atr_stop     = entry - (MAX_ATR_STOP * atr)
        stop         = max(five_bar_low, atr_stop)
        risk         = entry - stop

        if risk <= 0 or risk > entry * 0.12:
            return None

        size = min(int((ACCOUNT_SIZE * RISK_PER_TRADE) / risk), MAX_POSITION)
        if size <= 0:
            return None

        target   = entry + (risk * RR_RATIO)
        invested = round(entry * size, 0)

        # Candle quality label
        if candle_score >= 3:
            candle_label = "💪 Strong"
        elif candle_score >= 1:
            candle_label = "👍 Good"
        elif candle_score == 0:
            candle_label = "😐 Neutral"
        else:
            candle_label = "⚠️ Weak"

        return {
            "Symbol":   symbol,
            "Sector":   stock_sector,
            "Score":    score,
            "Candle":   candle_label,
            "Entry":    round(entry, 2),
            "Stop":     round(float(stop), 2),
            "Target":   round(float(target), 2),
            "Size":     size,
            "Invested": invested,
            "Risk$":    round(risk * size, 0),
            "Reward$":  round((float(target) - entry) * size, 0),
        }

    except Exception as e:
        print(f"  ⚠️ Error — {symbol}: {e}")
        return None

# =========================================
# STOCK UNIVERSE — 90+ stocks
# =========================================

sector_map = {
    # Semiconductors
    "NVDA":"SMH",  "AMD":"SMH",   "AVGO":"SMH",  "TSM":"SMH",
    "AMAT":"SMH",  "LRCX":"SMH",  "KLAC":"SMH",  "ASML":"SMH",
    "ARM":"SMH",   "MRVL":"SMH",  "ONTO":"SMH",  "ENTG":"SMH",
    "SMH":"SMH",   "SOXX":"SOXX",
    # AI Software
    "PLTR":"XLK",  "MSFT":"XLK",  "GOOGL":"XLC", "META":"XLC",
    "AMZN":"XLK",  "NOW":"XLK",   "CRM":"XLK",   "ADBE":"XLK",
    "INTU":"XLK",  "APP":"XLK",   "RBLX":"XLC",
    # Cybersecurity
    "CRWD":"XLK",  "PANW":"XLK",  "ZS":"XLK",
    "FTNT":"XLK",  "OKTA":"XLK",  "S":"XLK",     "CYBR":"XLK",
    # Cloud / Data
    "SNOW":"XLK",  "DDOG":"XLK",  "NET":"XLK",
    "ORCL":"XLK",  "MDB":"XLK",   "GTLB":"XLK",
    # EV / Mobility
    "TSLA":"XLY",  "UBER":"XLY",  "LYFT":"XLY",
    # Clean Energy
    "NEE":"XLU",   "FSLR":"TAN",  "ENPH":"TAN",
    "SEDG":"TAN",  "ICLN":"XLU",  "TAN":"TAN",
    # AI Infrastructure
    "SMCI":"SMH",  "ANET":"XLK",  "DELL":"XLK",
    "HPE":"XLK",   "VRT":"XLI",   "EQIX":"XLRE", "DLR":"XLRE",
    # Defense
    "LMT":"ITA",   "RTX":"ITA",   "NOC":"ITA",
    "GD":"ITA",    "KTOS":"ITA",  "LDOS":"ITA",
    "HII":"ITA",   "TDG":"ITA",   "ITA":"ITA",   "XAR":"XAR",
    # Space
    "RKLB":"ITA",  "ASTS":"ITA",  "LUNR":"ITA",
    "GE":"XLI",    "BA":"XLI",
    # Nuclear / Power Grid
    "CCJ":"URA",   "NXE":"URA",   "LEU":"URA",
    "SMR":"URA",   "OKLO":"URA",  "URA":"URA",   "URNM":"URA",
    "VST":"XLU",   "CEG":"XLU",   "ETN":"XLI",
    "GEV":"XLI",   "PWR":"XLI",   "ACHR":"XLI",
    # Biotech
    "LLY":"XBI",   "NVO":"XBI",   "VKTX":"XBI",
    "RXRX":"XBI",  "ROIV":"XBI",  "XBI":"XBI",
    # Quantum
    "IONQ":"XLK",  "RGTI":"XLK",  "QUBT":"XLK",
    # Financials
    "GS":"XLF",    "JPM":"XLF",   "V":"XLF",
    "MA":"XLF",    "PYPL":"XLF",  "AFRM":"XLF",
    "HOOD":"XLF",  "IBKR":"XLF",
    # Consumer
    "COST":"XLP",  "DECK":"XLY",  "ONON":"XLY",  "LULU":"XLY",
    # Commodities
    "GLD":"GLD",   "IAU":"GLD",   "SLV":"SLV",
    "COPX":"XLB",  "URA":"URA",   "WPM":"XLB",
    "GOLD":"XLB",  "MP":"XLB",
    # Materials
    "ALB":"XLB",   "SQM":"XLB",
}

stocks = list(dict.fromkeys(sector_map.keys()))

# =========================================
# MAIN
# =========================================

def run_agent():

    print(f"\n{'='*55}")
    print(f"🇺🇸 US Pro Scan — {datetime.now().strftime('%d %b %Y %H:%M:%S')}")
    print(f"{'='*55}")

    # Gate 1 — Market trend
    if not market_is_bullish():
        msg = (
            f"📉 S&P below EMA50/EMA200 — cash only\n"
            f"No new trades today.\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')}"
        )
        print(msg)
        send_telegram(msg)
        return

    # Gate 2 — Market breadth
    print("\nChecking market breadth...")
    breadth = get_market_breadth()
    if breadth < 40:
        msg = (
            f"⚠️ Market breadth WEAK — {breadth}% above EMA50\n"
            f"Only {breadth}% of S&P stocks are healthy.\n"
            f"Skipping new trades — too risky.\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')}"
        )
        print(msg)
        send_telegram(msg)
        return

    breadth_label = (
        "🟢 Strong" if breadth >= 60 else
        "🟡 Mixed"  if breadth >= 40 else
        "🔴 Weak"
    )

    # Gate 3 — Sector rotation
    print("\nChecking sector rotation...")
    hot_sectors = get_sector_rotation()

    # Download SPY benchmark
    spy_df = yf.download("^GSPC", period="1y", interval="1d", progress=False)
    if spy_df is None or spy_df.empty:
        print("⚠️ Failed to download S&P data")
        return
    spy_df = spy_df.dropna()
    spy_df.columns = spy_df.columns.get_level_values(0)

    picks            = []
    skipped_fund     = 0
    skipped_sector   = 0
    skipped_earnings = 0

    print(f"\nScanning {len(stocks)} stocks...")

    for stock in stocks:
        time.sleep(0.8)

        # Fundamental filter
        if not passes_fundamental_filter(stock):
            skipped_fund += 1
            continue

        # Sector long-term strength
        etf = sector_map.get(stock, "OTHER")
        if etf not in {"OTHER", "GLD", "SLV"}:
            if not sector_is_strong(etf):
                skipped_sector += 1
                continue

        # Earnings filter
        if is_near_earnings(stock):
            skipped_earnings += 1
            continue

        # Technical scan
        result = check_stock(stock, spy_df, hot_sectors)
        if result:
            print(
                f"  ✅ {result['Symbol']:6} [{result['Sector']:5}]"
                f" score:{result['Score']:3}"
                f" candle:{result['Candle']}"
                f" entry:${result['Entry']}"
            )
            picks.append(result)

    print(f"\n{'='*55}")
    print(f"Scanned  : {len(stocks)}")
    print(f"Fund ❌  : {skipped_fund}")
    print(f"Sector ❌: {skipped_sector}")
    print(f"Earnings❌: {skipped_earnings}")
    print(f"✅ Qualify: {len(picks)}")
    print(f"{'='*55}")

    if not picks:
        send_telegram(
            f"🔍 US Scan — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
            f"✅ Market bullish | Breadth: {breadth_label} ({breadth}%)\n"
            f"No high-quality setups found this scan.\n"
            f"Wait for next scan."
        )
        return

    picks = sorted(picks, key=lambda x: (x['Score'], x['Reward$']), reverse=True)

    # Hot sectors summary
    hot_str = ", ".join(sorted(hot_sectors)) if hot_sectors else "None"

    send_telegram(
        f"📊 US PRO SCAN — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
        f"{'='*32}\n"
        f"✅ S&P: Bullish (EMA50 + EMA200)\n"
        f"📈 Breadth: {breadth_label} ({breadth}%)\n"
        f"🔥 Hot Sectors: {hot_str}\n"
        f"🎯 {len(picks)} setup(s) found\n"
        f"Top {min(TOP_PICKS, len(picks))} picks below ↓\n"
        f"Trade what suits you — alerts only."
    )

    for pick in picks[:TOP_PICKS]:
        rr  = round(pick['Reward$'] / pick['Risk$'], 1) if pick['Risk$'] > 0 else 0
        msg = (
            f"{'='*32}\n"
            f"🚀 {pick['Symbol']}  [{pick['Sector']}]\n"
            f"Score   : {pick['Score']}/36\n"
            f"Candle  : {pick['Candle']}\n"
            f"Entry   : ${pick['Entry']}\n"
            f"Stop    : ${pick['Stop']}\n"
            f"Target  : ${pick['Target']}\n"
            f"Size    : {pick['Size']} shares\n"
            f"Invested: ${int(pick['Invested']):,}\n"
            f"Risk    : ${int(pick['Risk$']):,}\n"
            f"Reward  : ${int(pick['Reward$']):,}\n"
            f"RR      : 1:{rr}\n"
            f"{'='*32}"
        )
        send_telegram(msg)
        time.sleep(0.5)

# =========================================
# RUN
# =========================================

if __name__ == "__main__":
    print("🚀 US Professional Swing Trading Agent")
    run_agent()
    print("✅ Done")
