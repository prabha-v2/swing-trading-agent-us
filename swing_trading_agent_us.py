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

ACCOUNT_SIZE    = 30000        # Your actual capital
RISK_PER_TRADE  = 0.01         # 1% risk per trade = $300
RR_RATIO        = 2.5
MAX_ATR_STOP    = 3.0
MAX_POSITION    = 500
SCORE_THRESHOLD = 16           # Raised — stricter = better quality
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
            print(f"⚠️ Telegram error: {resp.status_code} — {resp.text}")
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")

# =========================================
# FUNDAMENTAL FILTER
# Removes weak companies before technical scan
# =========================================

SKIP_FUNDAMENTAL = {
    # ETFs and commodities — no fundamentals
    "SMH","SOXX","ITA","XAR","TAN","ICLN","URA","URNM",
    "ARKX","QTUM","XBI","XLU","XLI","GLD","IAU","SLV","COPX","UFO"
}

def passes_fundamental_filter(symbol):
    if symbol in SKIP_FUNDAMENTAL:
        return True
    try:
        info = yf.Ticker(symbol).info
        if not info:
            return True

        # Must not be deeply loss-making (EPS > -5 for US stocks)
        eps = info.get("trailingEps", None)
        if eps is not None and eps < -5:
            print(f"    ❌ {symbol}: EPS too negative ({eps:.1f}) — skip")
            return False

        # Debt/Equity < 300% (allows growth companies)
        de = info.get("debtToEquity", None)
        if de is not None and de > 300:
            print(f"    ❌ {symbol}: D/E too high ({de:.0f}) — skip")
            return False

        # Market cap > $1B (no micro caps)
        mktcap = info.get("marketCap", None)
        if mktcap is not None and mktcap < 1_000_000_000:
            print(f"    ❌ {symbol}: market cap too small — skip")
            return False

        # Revenue must be positive
        rev = info.get("totalRevenue", None)
        if rev is not None and rev <= 0:
            print(f"    ❌ {symbol}: zero/negative revenue — skip")
            return False

        return True
    except Exception:
        return True  # fail open

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
    latest       = df.iloc[-1]
    close        = float(latest['Close'])
    ema50        = float(latest['EMA50'])
    ema200       = float(latest['EMA200'])
    print(f"S&P: {close:.0f} | EMA50: {ema50:.0f} | EMA200: {ema200:.0f}")
    return close > ema50 and close > ema200

# =========================================
# SECTOR STRENGTH FILTER
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
# EARNINGS DATE FILTER
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
            print(f"⚠️ {symbol} earnings in {diff} days — skipping")
            return True
    except Exception:
        pass
    return False

# =========================================
# STOCK SCANNER — enhanced scoring
# =========================================

def check_stock(symbol, spy_df):
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False)

        if df is None or df.empty or len(df) < 250:
            return None

        df = df.dropna()
        df.columns = df.columns.get_level_values(0)

        if df.empty or len(df) < 250:
            return None

        # Price filter
        price = float(df['Close'].iloc[-1])
        if price < 5.0:
            return None

        # Liquidity — $2M+ daily dollar volume
        avg_dv = float(df['Close'].iloc[-20:].mean() * df['Volume'].iloc[-20:].mean())
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
        if rs >= 2:                                               score += 2
        if rs == 3:                                               score += 1  # bonus

        # ---- EMA TREND STACK ----
        if latest['EMA10']  > latest['EMA20']:                   score += 2
        if latest['EMA20']  > latest['EMA50']:                   score += 2
        if latest['EMA50']  > latest['EMA200']:                  score += 2
        if latest['Close']  > latest['EMA50']:                   score += 1
        if latest['Close']  > latest['EMA200']:                  score += 2

        # ---- RSI ----
        rsi = float(latest['RSI'])
        if prev['RSI'] < 50 and rsi > 50:                        score += 2  # RSI cross
        if 50 < rsi < 70:                                         score += 1  # healthy zone
        if rsi > 75:                                              score -= 1  # overbought penalty

        # ---- BREAKOUT ----
        if latest['Close'] > prev['HH20']:                        score += 2  # 20-day breakout
        ath_dist = latest['Close'] / latest['High52']
        if ath_dist > 0.90:                                       score += 2  # near 52W high
        elif ath_dist > 0.80:                                     score += 1

        # ---- VOLUME ----
        if latest['AvgVol'] > 0:
            rvol = latest['Volume'] / latest['AvgVol']
            if rvol > 2.0:                                        score += 2  # strong surge
            elif rvol > 1.5:                                      score += 1  # moderate

        # ---- ATR EXPANDING ----
        if latest['ATR'] > df['ATR'].iloc[-5]:                    score += 1

        # ---- ENTRY QUALITY ----
        dist = (latest['Close'] - latest['EMA20']) / latest['EMA20']
        if dist < 0.05:                                           score += 2  # tight pullback
        elif dist < 0.08:                                         score += 1  # acceptable
        elif dist > 0.20:                                         score -= 1  # too extended

        # ---- 60-DAY OUTPERFORMANCE ----
        if len(df) >= 60 and len(spy_df) >= 60:
            sr = float(df['Close'].squeeze().pct_change(60).iloc[-1])
            nr = float(spy_df['Close'].squeeze().pct_change(60).iloc[-1])
            if sr > nr * 1.5:                                     score += 2  # significantly outperforming
            elif sr > nr:                                         score += 1

        # ---- DISTRIBUTION PENALTY ----
        # Stock falling hard from recent high = institutional selling
        recent_high = df['High'].iloc[-10:].max()
        if latest['Close'] < recent_high * 0.85:
            score -= 2

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

        risk_amt = ACCOUNT_SIZE * RISK_PER_TRADE
        size     = min(int(risk_amt / risk), MAX_POSITION)

        if size <= 0:
            return None

        target = entry + (risk * RR_RATIO)

        return {
            "Symbol":  symbol,
            "Sector":  sector_map.get(symbol, "OTHER"),
            "Score":   score,
            "Entry":   round(entry, 2),
            "Stop":    round(float(stop), 2),
            "Target":  round(float(target), 2),
            "Size":    size,
            "Risk$":   round(risk * size, 0),
            "Reward$": round((float(target) - entry) * size, 0),
        }

    except Exception as e:
        print(f"  ⚠️ Error — {symbol}: {e}")
        return None

# =========================================
# STOCK UNIVERSE — expanded to 90+ stocks
# =========================================

sector_map = {

    # ---- AI / Semiconductors ----
    "NVDA": "SMH",  "AMD":  "SMH",  "AVGO": "SMH",  "TSM":  "SMH",
    "AMAT": "SMH",  "LRCX": "SMH",  "KLAC": "SMH",  "ASML": "SMH",
    "ARM":  "SMH",  "MRVL": "SMH",  "ONTO": "SMH",  "ENTG": "SMH",
    "SMH":  "SMH",  "SOXX": "SOXX",

    # ---- AI Software & Cloud ----
    "PLTR": "OTHER", "MSFT": "OTHER", "GOOGL": "OTHER",
    "META": "OTHER", "AMZN": "OTHER", "NOW":   "OTHER",
    "CRM":  "OTHER", "ADBE": "OTHER", "INTU":  "OTHER",
    "APP":  "OTHER", "TTWO": "OTHER", "RBLX":  "OTHER",

    # ---- Cybersecurity ----
    "CRWD": "OTHER", "PANW": "OTHER", "ZS":   "OTHER",
    "FTNT": "OTHER", "OKTA": "OTHER", "S":    "OTHER",
    "CYBR": "OTHER",

    # ---- Cloud / Data ----
    "SNOW": "OTHER", "DDOG": "OTHER", "NET":  "OTHER",
    "ORCL": "OTHER", "MDB":  "OTHER", "GTLB": "OTHER",

    # ---- EV / Robotics ----
    "TSLA": "OTHER", "UBER": "OTHER", "LYFT": "OTHER",

    # ---- Clean Energy / Solar ----
    "NEE":  "ICLN",  "FSLR": "TAN",  "ENPH": "TAN",
    "SEDG": "TAN",   "ICLN": "ICLN", "TAN":  "TAN",

    # ---- AI Infrastructure ----
    "SMCI": "SMH",  "ANET": "OTHER", "DELL": "OTHER",
    "HPE":  "OTHER","VRT":  "OTHER", "EQIX": "OTHER",
    "DLR":  "OTHER",

    # ---- Defense / Aerospace ----
    "LMT":  "ITA",  "RTX":  "ITA",  "NOC":  "ITA",
    "GD":   "ITA",  "KTOS": "ITA",  "LDOS": "ITA",
    "HII":  "ITA",  "TDG":  "ITA",  "ITA":  "ITA",
    "XAR":  "XAR",

    # ---- Space ----
    "RKLB": "ARKX", "ASTS": "ARKX", "LUNR": "ARKX",
    "GE":   "ARKX", "BA":   "ARKX",

    # ---- Nuclear / Power Grid ----
    "CCJ":  "URA",  "NXE":  "URA",  "LEU":  "URA",
    "SMR":  "URA",  "OKLO": "URA",  "URA":  "URA",
    "URNM": "URNM", "VST":  "XLU",  "CEG":  "XLU",
    "ETN":  "XLI",  "GEV":  "XLI",  "PWR":  "XLI",
    "ACHR": "XLI",

    # ---- Biotech / GLP-1 ----
    "LLY":  "XBI",  "NVO":  "XBI",  "VKTX": "XBI",
    "RXRX": "XBI",  "ROIV": "XBI",

    # ---- Quantum Computing ----
    "IONQ": "QTUM", "RGTI": "QTUM", "QUBT": "QTUM",

    # ---- Financials ----
    "GS":   "OTHER", "JPM":  "OTHER", "V":    "OTHER",
    "MA":   "OTHER", "PYPL": "OTHER", "AFRM": "OTHER",
    "HOOD": "OTHER", "IBKR": "OTHER",

    # ---- Consumer / Retail ----
    "AMZN": "OTHER", "COST": "OTHER", "DECK": "OTHER",
    "ONON": "OTHER", "LULU": "OTHER",

    # ---- ETFs ----
    "SMH":  "SMH",  "SOXX": "SOXX", "ITA":  "ITA",
    "XAR":  "XAR",  "UFO":  "ARKX",

    # ---- Commodities ----
    "GLD":  "GLD",  "IAU":  "GLD",  "SLV":  "SLV",
    "COPX": "COPX", "URA":  "URA",  "URNM": "URNM",
    "WPM":  "GLD",  "GOLD": "GLD",

    # ---- Materials ----
    "ALB":  "OTHER", "SQM": "OTHER", "MP":   "OTHER",
}

stocks = list(dict.fromkeys(sector_map.keys()))  # deduplicated

# =========================================
# MAIN — pure scanner
# =========================================

def run_agent():

    print(f"\n{'='*52}")
    print(f"🇺🇸 US Scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*52}")

    if not market_is_bullish():
        msg = (
            f"📉 S&P below EMA50/EMA200 — staying in cash\n"
            f"No new trades today.\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')} ET"
        )
        print(msg)
        send_telegram(msg)
        return

    spy_df = yf.download("^GSPC", period="1y", interval="1d", progress=False)
    if spy_df is None or spy_df.empty:
        print("⚠️ Failed to download S&P data")
        return
    spy_df = spy_df.dropna()
    spy_df.columns = spy_df.columns.get_level_values(0)

    picks              = []
    skipped_fund       = 0
    skipped_sector     = 0
    skipped_earnings   = 0

    for stock in stocks:
        time.sleep(0.8)

        # Step 1 — Fundamental filter
        if not passes_fundamental_filter(stock):
            skipped_fund += 1
            continue

        # Step 2 — Sector ETF strength
        etf = sector_map.get(stock)
        if etf and etf not in {"OTHER"}:
            if not sector_is_strong(etf):
                print(f"  {stock}: sector weak — skip")
                skipped_sector += 1
                continue

        # Step 3 — Earnings filter
        if is_near_earnings(stock):
            skipped_earnings += 1
            continue

        # Step 4 — Technical scan
        result = check_stock(stock, spy_df)
        if result:
            print(
                f"  ✅ {result['Symbol']} [{result['Sector']}]"
                f" score:{result['Score']}"
                f"  entry:${result['Entry']}"
            )
            picks.append(result)

    print(f"\nScanned: {len(stocks)} | Fund❌: {skipped_fund}"
          f" | Sector❌: {skipped_sector}"
          f" | Earnings❌: {skipped_earnings}"
          f" | ✅: {len(picks)}")

    if not picks:
        send_telegram(
            f"🔍 US Scan — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
            f"Market bullish but no high-quality setups found.\n"
            f"Wait for next scan."
        )
        return

    picks = sorted(picks, key=lambda x: (x['Score'], x['Reward$']), reverse=True)

    send_telegram(
        f"📊 US SCAN — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
        f"✅ S&P bullish (above EMA50 + EMA200)\n"
        f"🎯 {len(picks)} high-quality setup(s) found\n"
        f"Top {min(TOP_PICKS, len(picks))} picks below ↓\n"
        f"Trade what suits you — alerts only."
    )

    for pick in picks[:TOP_PICKS]:
        rr  = round(pick['Reward$'] / pick['Risk$'], 1) if pick['Risk$'] > 0 else 0
        msg = (
            f"{'='*30}\n"
            f"🚀 {pick['Symbol']}  [{pick['Sector']}]\n"
            f"Score  : {pick['Score']}/32\n"
            f"Entry  : ${pick['Entry']}\n"
            f"Stop   : ${pick['Stop']}\n"
            f"Target : ${pick['Target']}\n"
            f"Size   : {pick['Size']} shares\n"
            f"Risk   : ${int(pick['Risk$']):,}\n"
            f"Reward : ${int(pick['Reward$']):,}\n"
            f"RR     : 1:{rr}\n"
            f"{'='*30}"
        )
        send_telegram(msg)
        time.sleep(0.5)

# =========================================
# RUN
# =========================================

if __name__ == "__main__":
    print("🚀 US Swing Trading Agent — Enhanced")
    run_agent()
    print("✅ Done")
