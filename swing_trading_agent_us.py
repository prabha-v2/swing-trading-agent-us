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

ACCOUNT_SIZE = 30000           # Your actual capital
RISK_PER_TRADE = 0.01          # 1% risk per trade = $300
RR_RATIO = 2.5
MAX_ATR_STOP_MULTIPLIER = 3.0
MAX_POSITION_SIZE = 500
SCORE_THRESHOLD = 14
TOP_PICKS = 5                  # Max alerts per scan

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

# =========================================
# TELEGRAM
# =========================================

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        if not resp.ok:
            print(f"⚠️ Telegram error: {resp.status_code} — {resp.text}")
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")

# =========================================
# MARKET TREND FILTER (unchanged)
# =========================================

def market_is_bullish():
    df = yf.download("^GSPC", period="1y", interval="1d", progress=False)
    df = df.dropna()
    df.columns = df.columns.get_level_values(0)
    if df.empty:
        return False

    df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
    latest_close = float(df['Close'].iloc[-1])
    latest_ema   = float(df['EMA200'].iloc[-1])

    print(f"S&P Close: {latest_close:.2f} | EMA200: {latest_ema:.2f}")
    return latest_close > latest_ema

# =========================================
# SECTOR STRENGTH FILTER (unchanged)
# =========================================

def sector_is_strong(etf_symbol):
    df = yf.download(etf_symbol, period="1y", interval="1d", progress=False)
    df = df.dropna()
    df.columns = df.columns.get_level_values(0)
    if df.empty:
        return False

    df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
    return float(df['Close'].iloc[-1]) > float(df['EMA200'].iloc[-1])

# =========================================
# EARNINGS DATE FILTER (unchanged)
# =========================================

def is_near_earnings(symbol, days=5):
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is None or cal.empty:
            return False
        if 'Earnings Date' in cal.index:
            earn_date = cal.loc['Earnings Date'].iloc[0]
            if pd.isnull(earn_date):
                return False
            earn_date = pd.Timestamp(earn_date).date()
            today = datetime.now().date()
            diff = abs((earn_date - today).days)
            if diff <= days:
                print(f"⚠️ {symbol} earnings in {diff} days — skipping")
                return True
    except Exception as e:
        print(f"⚠️ Earnings check failed for {symbol}: {e}")
    return False

# =========================================
# STOCK CHECK (unchanged)
# =========================================

def check_stock(symbol, spy_df):

    df = yf.download(symbol, period="2y", interval="1d", progress=False)

    if df is None or df.empty or len(df) < 250:
        print(f"⚠️ Data error for {symbol}")
        return None

    df = df.dropna()
    df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 250:
        return None

    latest_close_price = float(df['Close'].iloc[-1])
    if latest_close_price < 5.0:
        print(f"⚠️ {symbol} price ${latest_close_price:.2f} below $5 — skipping")
        return None

    avg_dollar_vol = float(df['Close'].iloc[-20:].mean() * df['Volume'].iloc[-20:].mean())
    if avg_dollar_vol < 1_000_000:
        print(f"⚠️ {symbol} avg dollar vol ${avg_dollar_vol:,.0f} too low — skipping")
        return None

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
    ).combine(
        abs(df['High'] - df['Close'].shift(1)), max
    ).combine(
        abs(df['Low'] - df['Close'].shift(1)), max
    )
    df['ATR'] = df['TR'].rolling(14).mean()

    stock_3m  = df['Close'].pct_change(63).iloc[-1]
    stock_6m  = df['Close'].pct_change(126).iloc[-1]
    stock_12m = df['Close'].pct_change(252).iloc[-1]
    spy_3m    = spy_df['Close'].pct_change(63).iloc[-1]
    spy_6m    = spy_df['Close'].pct_change(126).iloc[-1]
    spy_12m   = spy_df['Close'].pct_change(252).iloc[-1]

    rs_score = 0
    if stock_3m  > spy_3m:  rs_score += 1
    if stock_6m  > spy_6m:  rs_score += 1
    if stock_12m > spy_12m: rs_score += 1

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    score = 0
    if rs_score >= 2:                                         score += 2
    if latest['EMA10'] > latest['EMA20']:                     score += 2
    if latest['EMA50'] > latest['EMA200']:                    score += 2
    if latest['EMA20'] > latest['EMA50']:                     score += 2
    if latest['Close'] > latest['EMA50']:                     score += 2
    if latest['Close'] > latest['EMA200']:                    score += 2
    if prev['RSI'] < 50 and latest['RSI'] > 50:               score += 2
    if latest['Close'] > prev['HH20']:                        score += 2
    if latest['Close'] / latest['High52'] > 0.85:             score += 2
    if latest['AvgVol'] > 0:
        if latest['Volume'] / latest['AvgVol'] > 1.5:        score += 2
    if latest['ATR'] > df['ATR'].iloc[-5]:                    score += 2
    distance_from_ema20 = (latest['Close'] - latest['EMA20']) / latest['EMA20']
    if distance_from_ema20 < 0.05:                            score += 2

    if len(df) < 60:
        stock_return = 0
    else:
        stock_return = float(df['Close'].squeeze().pct_change(60).iloc[-1])
    if spy_df is None or spy_df.empty or len(spy_df) < 60:
        spy_return = 0
    else:
        spy_return = float(spy_df['Close'].squeeze().pct_change(60).iloc[-1])
    if stock_return > spy_return:
        score += 2

    if score < SCORE_THRESHOLD:
        return None

    entry = float(latest['Close'])
    atr   = float(latest['ATR'])

    five_bar_low = float(df['Low'].iloc[-5:].min())
    atr_stop     = entry - (MAX_ATR_STOP_MULTIPLIER * atr)
    stop         = max(five_bar_low, atr_stop)
    risk         = entry - stop

    if risk <= 0 or risk > entry * 0.15:
        return None

    risk_amount   = ACCOUNT_SIZE * RISK_PER_TRADE
    position_size = min(int(risk_amount / risk), MAX_POSITION_SIZE)

    if position_size <= 0:
        return None

    target = entry + (risk * RR_RATIO)

    return {
        "Symbol":  symbol,
        "Entry":   round(entry, 2),
        "Stop":    round(float(stop), 2),
        "Target":  round(float(target), 2),
        "Size":    position_size,
        "Score":   score,
        "Sector":  sector_map.get(symbol, "OTHER"),
        "Reward":  round(float(target) - entry, 2),
        "Risk$":   round(risk * position_size, 0),
        "Reward$": round((float(target) - entry) * position_size, 0),
    }

# =========================================
# STOCK UNIVERSE (unchanged)
# =========================================

sector_map = {
    "NVDA": "SMH", "AMD": "SMH", "AVGO": "SMH", "TSM": "SMH",
    "AMAT": "SMH", "LRCX": "SMH", "KLAC": "SMH", "ASML": "SMH",
    "ARM":  "SMH", "SMH":  "SMH", "SOXX": "SOXX",
    "FSLR": "TAN", "TAN": "TAN", "ICLN": "ICLN", "NEE": "ICLN",
    "ENPH": "TAN", "SEDG": "TAN",
    "LMT": "ITA", "RTX": "ITA", "NOC": "ITA", "GD":   "ITA",
    "BA":  "ITA", "ITA": "ITA", "XAR": "XAR",
    "KTOS":"ITA", "LDOS":"ITA",
    "RKLB": "ARKX", "LUNR": "ARKX", "ASTS": "ARKX",
    "CCJ":  "URA", "NXE":  "URA", "LEU":  "URA",
    "SMR":  "URA", "OKLO": "URA", "URA":  "URA", "URNM": "URNM",
    "VST":  "XLU", "CEG":  "XLU", "ETN":  "XLI",
    "GEV":  "XLI", "PWR":  "XLI",
    "LLY":  "XBI", "NVO":  "XBI", "VKTX": "XBI",
    "IONQ": "QTUM", "RGTI": "QTUM", "QUBT": "QTUM",
    "GLD": "GLD", "IAU": "GLD", "SLV": "SLV", "COPX": "COPX",
}

stocks = [
    "NVDA", "AMD", "AVGO", "TSM", "ASML", "ARM", "AMAT", "LRCX", "KLAC",
    "PLTR", "MSFT", "GOOGL", "META", "AMZN", "NOW", "CRM", "ADBE", "INTU",
    "CRWD", "PANW", "ZS", "FTNT", "OKTA",
    "SNOW", "DDOG", "NET", "ORCL",
    "TSLA",
    "NEE", "FSLR", "ICLN", "TAN", "ENPH", "SEDG",
    "SMCI", "ANET", "DELL", "HPE",
    "LMT", "RTX", "NOC", "GD", "KTOS", "LDOS",
    "RKLB", "LUNR", "ASTS",
    "CCJ", "NXE", "LEU", "SMR", "OKLO", "VST", "CEG", "ETN", "GEV", "PWR",
    "LLY", "NVO", "VKTX",
    "IONQ", "RGTI", "QUBT",
    "SMH", "SOXX", "ITA", "XAR",
    "GLD", "IAU", "SLV", "COPX", "URA", "URNM",
    "ALB", "SQM",
]

# =========================================
# MAIN — pure scanner, no trade tracking
# =========================================

def run_agent():

    print(f"\n{'='*50}")
    print(f"Scan started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    if not market_is_bullish():
        msg = "📉 Market below EMA200 — staying in cash. No trades today."
        print(msg)
        send_telegram(msg)
        return

    spy_df = yf.download("^GSPC", period="1y", interval="1d", progress=False)
    if spy_df is None or spy_df.empty:
        print("⚠️ Failed to download S&P data")
        return
    spy_df = spy_df.dropna()
    spy_df.columns = spy_df.columns.get_level_values(0)

    picks = []

    for stock in stocks:
        time.sleep(1)

        if stock in sector_map:
            if not sector_is_strong(sector_map[stock]):
                print(f"  {stock}: sector weak — skip")
                continue

        if is_near_earnings(stock):
            continue

        result = check_stock(stock, spy_df)
        if result:
            print(f"  ✅ {stock} qualifies — score {result['Score']}/28")
            picks.append(result)

    print(f"\nScan done. {len(picks)} stocks qualified.")

    if not picks:
        send_telegram(
            f"🔍 Scan complete — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
            f"Market bullish but no stocks meet all criteria right now.\n"
            f"Nothing to act on — wait for next scan."
        )
        return

    picks = sorted(picks, key=lambda x: (x['Score'], x['Reward$']), reverse=True)

    send_telegram(
        f"📊 SCAN RESULTS — {datetime.now().strftime('%d %b %Y %H:%M')}\n"
        f"✅ Market bullish (S&P above EMA200)\n"
        f"🎯 {len(picks)} stock(s) qualify — top {min(TOP_PICKS, len(picks))} below\n"
        f"Trade what suits you — these are alerts only."
    )

    for pick in picks[:TOP_PICKS]:
        msg = (
            f"{'='*28}\n"
            f"🚀 {pick['Symbol']}  [{pick['Sector']}]\n"
            f"Score  : {pick['Score']}/28\n"
            f"Entry  : ${pick['Entry']}\n"
            f"Stop   : ${pick['Stop']}\n"
            f"Target : ${pick['Target']}\n"
            f"Size   : {pick['Size']} shares\n"
            f"Risk   : ${int(pick['Risk$']):,}\n"
            f"Reward : ${int(pick['Reward$']):,}\n"
            f"{'='*28}"
        )
        send_telegram(msg)
        time.sleep(0.5)

# =========================================
# RUN
# =========================================

if __name__ == "__main__":
    print("🚀 Swing Trading Agent Started")
    run_agent()
    print("✅ Done")
