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

ACCOUNT_SIZE     = 30000
RISK_PER_TRADE   = 0.01
RR_RATIO         = 2.5
MAX_ATR_STOP     = 3.0
MAX_POSITION_PCT = 0.15          # max 15% of account per trade ($4,500) — replaces share cap
SCORE_THRESHOLD  = 22            # raised from 18 — requires stronger multi-indicator confluence
TOP_PICKS        = 5
MAX_PER_SECTOR   = 2

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")

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
        return 50
    pct = round((above_50 / total) * 100, 1)
    print(f"Market Breadth: {above_50}/{total} above EMA50 = {pct}%")
    return pct

# =========================================
# SECTOR ROTATION
# =========================================

def get_sector_rotation():
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
    hot_sectors = set()
    sector_perf = {}
    for etf, name in sector_etfs.items():
        try:
            df = yf.download(etf, period="3mo", interval="1d", progress=False)
            df = df.dropna()
            df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 21:
                continue
            ret_1w = float(df['Close'].pct_change(5).iloc[-1])
            ret_1m = float(df['Close'].pct_change(21).iloc[-1])
            avg_r  = float(df['Volume'].iloc[-5:].mean())
            avg_o  = float(df['Volume'].iloc[-21:-5].mean())
            vol_tr = avg_r / avg_o if avg_o > 0 else 1
            sector_perf[etf] = {
                "name": name, "ret_1w": ret_1w,
                "ret_1m": ret_1m, "vol_trend": vol_tr
            }
            time.sleep(0.3)
        except Exception:
            continue
    if not sector_perf:
        return hot_sectors
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
        print(f"  {flag} {etf:5} {d['name']:20} 1W:{d['ret_1w']:+.1%} 1M:{d['ret_1m']:+.1%}")
    return hot_sectors

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

def is_near_earnings(symbol, days=12):  # extended from 5 to 12 — stocks move unpredictably 2wks before earnings
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

def check_stock(symbol, spy_df, hot_sectors, risk_pct=RISK_PER_TRADE):
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

        # ---- ADX (Average Directional Index — trend strength) ----
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        adx_val   = float(df['ADX'].iloc[-1]) if not pd.isna(df['ADX'].iloc[-1]) else 20.0

        # ---- Bollinger Band Squeeze ----
        # When bands are at their tightest in 60 days, a big move is coiling
        df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
        df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']
        bb_squeeze = False
        if len(df) >= 60:
            pct20      = float(df['BB_width'].iloc[-60:].quantile(0.20))
            bb_squeeze = float(df['BB_width'].iloc[-1]) < pct20

        # ---- OBV (On-Balance Volume) ----
        df['OBV']       = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['OBV_EMA20'] = ta.trend.ema_indicator(df['OBV'], window=20)
        obv_rising      = float(df['OBV'].iloc[-1]) > float(df['OBV_EMA20'].iloc[-1])
        obv_slope       = float(df['OBV'].iloc[-1]) - float(df['OBV'].iloc[-10])  # slope avoids pct_change bug on negative OBV
        obv_trend_pos   = obv_slope > 0

        # ---- MACD ----
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

        # ---- Relative strength vs SPY ----
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

        # ---- RELATIVE STRENGTH ----
        if rs >= 2:  score += 2
        if rs == 3:  score += 1

        # ---- EMA TREND STACK ----
        if latest['EMA10']  > latest['EMA20']:  score += 2
        if latest['EMA20']  > latest['EMA50']:  score += 2
        if latest['EMA50']  > latest['EMA200']: score += 2
        if latest['Close']  > latest['EMA50']:  score += 1
        if latest['Close']  > latest['EMA200']: score += 2

        # ---- RSI (improved thresholds) ----
        rsi      = float(latest['RSI'])
        rsi_prev = float(prev['RSI'])
        if rsi > 60 and rsi_prev < 60:         score += 2  # momentum breakout above 60
        elif 55 < rsi < 75:                     score += 2  # strong momentum zone
        elif 40 < rsi < 55 and rsi > rsi_prev: score += 1  # pullback reset recovering in uptrend
        elif rsi > 80:                          score -= 2  # severely overbought — likely to stall
        elif rsi < 40:                          score -= 1  # weak momentum

        # ---- BREAKOUT ----
        if latest['Close'] > prev['HH20']:  score += 2
        ath_dist = latest['Close'] / latest['High52']
        if ath_dist > 0.90:                 score += 2
        elif ath_dist > 0.80:              score += 1

        # ---- VOLUME ----
        rvol = 0.0
        if latest['AvgVol'] > 0:
            rvol = latest['Volume'] / latest['AvgVol']
            if rvol > 2.0:    score += 2
            elif rvol > 1.5:  score += 1

        # ---- ATR EXPANDING ----
        if latest['ATR'] > df['ATR'].iloc[-5]:  score += 1

        # ---- ENTRY QUALITY ----
        dist = (latest['Close'] - latest['EMA20']) / latest['EMA20']
        if dist < 0.05:    score += 2
        elif dist < 0.08:  score += 1
        elif dist > 0.20:  score -= 1

        # ---- 60-DAY OUTPERFORMANCE ----
        if len(df) >= 60 and len(spy_df) >= 60:
            sr = float(df['Close'].squeeze().pct_change(60).iloc[-1])
            nr = float(spy_df['Close'].squeeze().pct_change(60).iloc[-1])
            if sr > nr * 1.5:  score += 2
            elif sr > nr:      score += 1

        # ---- DISTRIBUTION PENALTY ----
        recent_high = df['High'].iloc[-10:].max()
        if latest['Close'] < recent_high * 0.85:  score -= 2

        # ---- OBV SCORING (slope-based — pct_change is invalid on negative OBV) ----
        if obv_rising and obv_trend_pos:               score += 2  # institutions buying
        elif obv_rising:                                score += 1  # OBV above its EMA only
        elif not obv_trend_pos and not obv_rising:      score -= 2  # heavy distribution

        # ---- MACD SCORING ----
        if macd_crossed_up:                             score += 2  # fresh crossover = best signal
        elif macd_above_sig and macd_hist_rising:       score += 2  # above signal + accelerating
        elif macd_above_sig:                            score += 1  # above signal only
        elif not macd_above_sig and macd_hist_now < 0: score -= 1  # weak momentum

        # ---- ADX SCORING (trend strength filter) ----
        if adx_val > 30:    score += 2  # strong trend — high conviction
        elif adx_val > 20:  score += 1  # developing trend
        elif adx_val < 15:  score -= 2  # choppy sideways action — penalize hard

        # ---- BOLLINGER BAND SQUEEZE ----
        if bb_squeeze:  score += 3  # volatility coiling — explosive move imminent

        # ---- CANDLE QUALITY ----
        candle_score = candle_quality_score(df)
        score += candle_score

        # ---- SECTOR ROTATION BONUS ----
        stock_sector = sector_map.get(symbol, "OTHER")
        if stock_sector in hot_sectors:  score += 2

        if score < SCORE_THRESHOLD:
            return None

        # ---- POSITION SIZING ----
        entry          = float(latest['Close'])
        atr            = float(latest['ATR'])
        ten_bar_low    = float(df['Low'].iloc[-10:].min())  # wider structural stop vs 5-bar
        atr_stop       = entry - (MAX_ATR_STOP * atr)
        stop           = max(ten_bar_low, atr_stop)
        risk           = entry - stop

        if risk <= 0 or risk > entry * 0.12:
            return None

        # Cap by both risk% and max dollar allocation (prevents 500 shares of $200 stock)
        max_by_dollars = int((ACCOUNT_SIZE * MAX_POSITION_PCT) / entry)
        size           = min(int((ACCOUNT_SIZE * risk_pct) / risk), max_by_dollars)
        if size <= 0:
            return None

        target   = entry + (risk * RR_RATIO)
        invested = round(entry * size, 0)
        acct_pct = round((invested / ACCOUNT_SIZE) * 100, 1)

        # ---- SETUP TYPE ----
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

        # ---- LABELS ----
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

        return {
            "Symbol":   symbol,
            "Sector":   stock_sector,
            "Score":    score,
            "Setup":    setup_type,
            "Candle":   candle_label,
            "MACD":     macd_label,
            "OBV":      obv_label,
            "ADX":      adx_label,
            "Squeeze":  "Yes" if bb_squeeze else "No",
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
# STOCK UNIVERSE
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
    "EXPE":"XLY",  "CELH":"XLY",
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

stocks = list(dict.fromkeys(sector_map.keys()))

# =========================================
# MAIN
# =========================================

def run_agent():

    print(f"\n{'='*55}")
    print(f"US Pro Scan — {datetime.now().strftime('%d %b %Y %H:%M:%S')}")
    print(f"{'='*55}")

    if not market_is_bullish():
        msg = (
            f"📉 S&P below EMA50/EMA200 — cash only\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')}"
        )
        send_telegram(msg)
        return

    # ---- VIX Regime Filter ----
    vix = get_vix()
    if vix > 30:
        send_telegram(
            f"⚠️ VIX={vix:.0f} — extreme fear mode\n"
            f"Swing setups fail at high rates when VIX >30.\n"
            f"Skipping scan — stay in cash.\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')}"
        )
        return
    # Half position size when VIX is elevated (25-30) to reduce exposure in choppy markets
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
    hot_sectors = get_sector_rotation()

    spy_df = yf.download("^GSPC", period="1y", interval="1d", progress=False)
    if spy_df is None or spy_df.empty:
        return
    spy_df = spy_df.dropna()
    spy_df.columns = spy_df.columns.get_level_values(0)

    picks         = []
    sector_counts = {}
    skipped_fund  = 0
    skipped_sec   = 0
    skipped_earn  = 0
    skipped_corr  = 0

    print(f"\nScanning {len(stocks)} stocks...")

    for stock in stocks:
        time.sleep(0.8)

        if not passes_fundamental_filter(stock):
            skipped_fund += 1
            continue

        etf = sector_map.get(stock, "OTHER")
        if etf not in {"OTHER","GLD","SLV"}:
            if not sector_is_strong(etf):
                skipped_sec += 1
                continue

        if is_near_earnings(stock):
            skipped_earn += 1
            continue

        result = check_stock(stock, spy_df, hot_sectors, risk_pct=effective_risk)

        if result:
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
    print(f"Scanned:{len(stocks)} Fund❌:{skipped_fund} "
          f"Sector❌:{skipped_sec} Earn❌:{skipped_earn} "
          f"Corr❌:{skipped_corr} ✅:{len(picks)}")
    print(f"{'='*55}")

    if not picks:
        send_telegram(
            f"🔍 US Scan — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
            f"Market bullish | Breadth:{breadth_label} ({breadth}%)\n"
            f"No high-quality setups found — wait for next scan."
        )
        return

    picks   = sorted(picks, key=lambda x: (x['Score'], x['Reward$']), reverse=True)
    hot_str = ", ".join(sorted(hot_sectors)) if hot_sectors else "None"

    send_telegram(
        f"📊 US PRO SCAN — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
        f"{'='*34}\n"
        f"S&P    : Bullish (EMA50 + EMA200)\n"
        f"VIX    : {vix_label}\n"
        f"Breadth: {breadth_label} ({breadth}%)\n"
        f"Hot    : {hot_str}\n"
        f"Setups : {len(picks)} found\n"
        f"Top {min(TOP_PICKS,len(picks))} picks below — alerts only."
    )

    for pick in picks[:TOP_PICKS]:
        rr  = round(pick['Reward$'] / pick['Risk$'], 1) if pick['Risk$'] > 0 else 0
        msg = (
            f"{'='*34}\n"
            f"🚀 {pick['Symbol']}  [{pick['Sector']}]\n"
            f"Setup   : {pick['Setup']}\n"
            f"Score   : {pick['Score']}\n"
            f"Candle  : {pick['Candle']}\n"
            f"MACD    : {pick['MACD']}\n"
            f"OBV     : {pick['OBV']}\n"
            f"ADX     : {pick['ADX']}\n"
            f"Squeeze : {pick['Squeeze']}\n"
            f"Entry   : ${pick['Entry']}\n"
            f"Stop    : ${pick['Stop']}\n"
            f"Target  : ${pick['Target']}\n"
            f"Size    : {pick['Size']} shares\n"
            f"Invested: ${int(pick['Invested']):,} ({pick['AcctPct']}%)\n"
            f"Risk    : ${int(pick['Risk$']):,}\n"
            f"Reward  : ${int(pick['Reward$']):,}\n"
            f"RR      : 1:{rr}\n"
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
    # 12:00 UTC (pre-open) to 21:30 UTC (includes the post-close EOD scan at 21:00 UTC)
    return (12 * 60) <= time_val <= (21 * 60 + 30)

if __name__ == "__main__":
    print("🚀 US Professional Swing Trading Agent")
    print(f"Started at {datetime.utcnow().strftime('%H:%M UTC')}")

    if is_market_hours():
        run_agent()
    else:
        print(f"Outside market hours — waiting...")

    while True:
        time.sleep(30 * 60)
        if is_market_hours():
            run_agent()
        else:
            print(f"Market closed — exiting.")
            break

    print("✅ Done — market closed.")
