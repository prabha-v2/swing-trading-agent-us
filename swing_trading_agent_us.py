import yfinance as yf
import pandas as pd
import ta
import time
import os
import requests
from datetime import datetime, timedelta

# =========================================
# SETTINGS
# =========================================

ACCOUNT_SIZE = 100000
RISK_PER_TRADE = 0.01          # 1% risk per trade = $1,000
RR_RATIO = 2.5                 # Risk:Reward ratio
TRAIL_ATR_MULTIPLIER = 2.0     # FIX: was 1.5 (too tight), now 2.0 for swing breathing room
MAX_ATR_STOP_MULTIPLIER = 3.0  # NEW: max stop = 3x ATR from entry, prevents huge position sizes
MAX_POSITION_SIZE = 500        # NEW: cap at 500 shares to prevent illiquid/penny stock blowups
MAX_OPEN_TRADES = 5
MAX_SAME_SECTOR = 2            # NEW: max 2 stocks from same sector to avoid concentration risk
SCORE_THRESHOLD = 14           # Minimum score to qualify for a trade

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8606993261:AAHyDbp0_aGTOoZZPRLI7CF91MyUHjOOb2c")
CHAT_ID = os.environ.get("CHAT_ID", "8537564769")

OPEN_TRADES_FILE = "open_trades.csv"
TRADE_LOG_FILE = "trade_log.csv"

# =========================================
# TELEGRAM  (FIX: added error handling)
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
    latest_close = df['Close'].iloc[-1]
    latest_ema = df['EMA200'].iloc[-1]

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
    return df['Close'].iloc[-1] > df['EMA200'].iloc[-1]

# =========================================
# NEW: EARNINGS DATE FILTER
# Avoids entering within 5 days of earnings
# =========================================

def is_near_earnings(symbol, days=5):
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is None or cal.empty:
            return False
        # calendar returns a DataFrame with 'Earnings Date' row
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
# STOCK CHECK  (all fixes applied)
# =========================================

def check_stock(symbol, spy_df):

    df = yf.download(symbol, period="2y", interval="1d", progress=False)

    if df is None or df.empty or len(df) < 250:
        print(f"⚠️ Data error for {symbol}")
        return None

    df = df.dropna()
    df.columns = df.columns.get_level_values(0)  # FIX: only called once now

    if df.empty or len(df) < 250:
        return None

    # NEW: minimum price filter — skip penny stocks under $5
    latest_close_price = float(df['Close'].iloc[-1])
    if latest_close_price < 5.0:
        print(f"⚠️ {symbol} price ${latest_close_price:.2f} below $5 minimum — skipping")
        return None

    # NEW: minimum average dollar volume filter — need $1M+ daily liquidity
    avg_dollar_vol = float(df['Close'].iloc[-20:].mean() * df['Volume'].iloc[-20:].mean())
    if avg_dollar_vol < 1_000_000:
        print(f"⚠️ {symbol} avg dollar volume ${avg_dollar_vol:,.0f} too low — skipping")
        return None

    df['EMA10']  = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA20']  = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
    df['RSI']    = ta.momentum.rsi(df['Close'], window=14)
    df['AvgVol'] = df['Volume'].rolling(20).mean()

    # FIX: use 20-day high breakout instead of 5-day (more meaningful for swing trades)
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

    # Relative Strength vs Market
    stock_3m  = df['Close'].pct_change(63).iloc[-1]
    stock_6m  = df['Close'].pct_change(126).iloc[-1]
    stock_12m = df['Close'].pct_change(252).iloc[-1]

    spy_3m  = spy_df['Close'].pct_change(63).iloc[-1]
    spy_6m  = spy_df['Close'].pct_change(126).iloc[-1]
    spy_12m = spy_df['Close'].pct_change(252).iloc[-1]

    rs_score = 0
    if stock_3m  > spy_3m:  rs_score += 1
    if stock_6m  > spy_6m:  rs_score += 1
    if stock_12m > spy_12m: rs_score += 1

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    score = 0

    if rs_score >= 2:
        score += 2

    if latest['EMA10'] > latest['EMA20']:
        score += 2
    if latest['EMA50'] > latest['EMA200']:
        score += 2
    if latest['EMA20'] > latest['EMA50']:
        score += 2
    if latest['Close'] > latest['EMA50']:
        score += 2
    if latest['Close'] > latest['EMA200']:
        score += 2
    if prev['RSI'] < 50 and latest['RSI'] > 50:
        score += 2

    # FIX: 20-day high breakout instead of 5-day
    if latest['Close'] > prev['HH20']:
        score += 2

    distance_from_high = latest['Close'] / latest['High52']
    if distance_from_high > 0.85:
        score += 2

    if latest['AvgVol'] > 0:
        relative_volume = latest['Volume'] / latest['AvgVol']
    else:
        relative_volume = 0
    if relative_volume > 1.5:
        score += 2

    if latest['ATR'] > df['ATR'].iloc[-5]:
        score += 2

    # FIX: tightened from 8% to 5% for cleaner pullback entries
    distance_from_ema20 = (latest['Close'] - latest['EMA20']) / latest['EMA20']
    if distance_from_ema20 < 0.05:
        score += 2

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

    entry = latest['Close']
    atr   = latest['ATR']

    # FIX: use max of (5-bar low, entry - 3×ATR) so stop is never too far away
    five_bar_low = df['Low'].iloc[-5:].min()
    atr_stop     = entry - (MAX_ATR_STOP_MULTIPLIER * atr)
    stop         = max(five_bar_low, atr_stop)

    risk = entry - stop

    if risk <= 0 or risk > entry * 0.15:   # also cap risk at 15% of entry price
        return None

    risk_amount   = ACCOUNT_SIZE * RISK_PER_TRADE
    position_size = int(risk_amount / risk)

    # FIX: apply maximum position size cap
    position_size = min(position_size, MAX_POSITION_SIZE)

    if position_size <= 0:
        return None

    target = entry + (risk * RR_RATIO)

    return {
        "Symbol":  symbol,
        "Entry":   round(float(entry), 2),
        "Stop":    round(float(stop), 2),
        "Target":  round(float(target), 2),
        "Size":    position_size,
        "Score":   score,
        "Sector":  sector_map.get(symbol, "OTHER")
    }

# =========================================
# STOCK UNIVERSE  (expanded with new themes)
# =========================================

sector_map = {
    # Semiconductors / AI chips
    "NVDA": "SMH", "AMD": "SMH", "AVGO": "SMH", "TSM": "SMH",
    "AMAT": "SMH", "LRCX": "SMH", "KLAC": "SMH", "ASML": "SMH",
    "ARM":  "SMH", "SMH":  "SMH", "SOXX": "SOXX",

    # Clean Energy / Solar
    "FSLR": "TAN", "TAN": "TAN", "ICLN": "ICLN", "NEE": "ICLN",
    "ENPH": "TAN", "SEDG": "TAN",

    # Defense / Space
    "LMT": "ITA", "RTX": "ITA", "NOC": "ITA", "GD":   "ITA",
    "BA":  "ITA", "ITA": "ITA", "XAR": "XAR",
    "KTOS":"ITA", "LDOS":"ITA",

    # Space (dedicated)
    "RKLB":  "ARKX", "LUNR": "ARKX", "ASTS": "ARKX",

    # Nuclear / Power grid
    "CCJ":  "URA",  "NXE":  "URA",  "LEU":  "URA",
    "SMR":  "URA",  "OKLO": "URA",  "URA":  "URA",  "URNM": "URNM",
    "VST":  "XLU",  "CEG":  "XLU",  "ETN":  "XLI",
    "GEV":  "XLI",  "PWR":  "XLI",

    # Biotech / GLP-1
    "LLY":  "XBI",  "NVO":  "XBI",  "VKTX": "XBI",

    # Quantum computing
    "IONQ": "QTUM", "RGTI": "QTUM", "QUBT": "QTUM",



    # Commodities
    "GLD": "GLD", "IAU": "GLD", "SLV": "SLV",
    "COPX":"COPX",
}

stocks = [

    # --- AI / Semiconductors ---
    "NVDA", "AMD", "AVGO", "TSM", "ASML", "ARM",
    "AMAT", "LRCX", "KLAC",

    # --- AI Software ---
    "PLTR", "MSFT", "GOOGL", "META", "AMZN",
    "NOW",  "CRM",  "ADBE",  "INTU",

    # --- Cybersecurity ---
    "CRWD", "PANW", "ZS", "FTNT", "OKTA",

    # --- Cloud / Data ---
    "SNOW", "DDOG", "NET", "ORCL",

    # --- EV / Robotics ---
    "TSLA",

    # --- Clean Energy / Solar ---
    "NEE", "FSLR", "ICLN", "TAN", "ENPH", "SEDG",

    # --- AI Infrastructure ---
    "SMCI", "ANET", "DELL", "HPE",

    # --- Defense / Aerospace ---
    "LMT", "RTX", "NOC", "GD", "KTOS", "LDOS",

    # --- Space (NEW theme) ---
    "RKLB", "LUNR", "ASTS",

    # --- Nuclear / Power Grid (NEW theme) ---
    "CCJ", "NXE", "LEU", "SMR", "OKLO",
    "VST", "CEG", "ETN", "GEV", "PWR",

    # --- Biotech / GLP-1 (NEW theme) ---
    "LLY", "NVO", "VKTX",

    # --- Quantum Computing (NEW theme, speculative) ---
    "IONQ", "RGTI", "QUBT",

    # --- ETFs ---
    "SMH", "SOXX", "ITA", "XAR",

    # --- Commodities ---
    "GLD", "IAU", "SLV", "COPX", "URA", "URNM",

    # --- Materials ---
    "ALB", "SQM",
]

# =========================================
# TRADE MANAGEMENT
# =========================================

def load_open_trades():
    if os.path.exists(OPEN_TRADES_FILE):
        df = pd.read_csv(OPEN_TRADES_FILE)
        # ensure Sector column exists in older CSV files
        if "Sector" not in df.columns:
            df["Sector"] = "OTHER"
        return df
    return pd.DataFrame(columns=["Symbol", "Entry", "Stop", "Target", "Size", "Score", "Sector"])

def save_open_trades(df):
    df.to_csv(OPEN_TRADES_FILE, index=False)

def log_trade(symbol, entry, exit_price, size):
    pnl = (exit_price - entry) * size
    record = pd.DataFrame([{
        "Date":   datetime.now(),
        "Symbol": symbol,
        "Entry":  entry,
        "Exit":   exit_price,
        "Size":   size,
        "PnL":    round(pnl, 2)
    }])
    if os.path.exists(TRADE_LOG_FILE):
        record.to_csv(TRADE_LOG_FILE, mode='a', header=False, index=False)
    else:
        record.to_csv(TRADE_LOG_FILE, index=False)

def manage_trades():

    trades = load_open_trades()
    if trades.empty:
        return

    updated = []

    for _, trade in trades.iterrows():

        symbol = trade["Symbol"]

        # FIX: use 6 months instead of 1 month for reliable ATR calculation
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)

        if df is None or df.empty:
            print(f"⚠️ Data download failed for {symbol}")
            updated.append(trade)   # keep trade, retry next run
            continue

        df = df.dropna()
        df.columns = df.columns.get_level_values(0)

        if df.empty:
            updated.append(trade)
            continue

        latest = df.iloc[-1]

        atr = ta.volatility.average_true_range(
            df['High'], df['Low'], df['Close'], window=14
        ).iloc[-1]

        new_stop = latest['Close'] - (atr * TRAIL_ATR_MULTIPLIER)
        stop     = max(trade["Stop"], new_stop)

        # STOP LOSS HIT
        if latest['Low'] <= stop:
            msg = (
                f"❌ STOP HIT: {symbol}\n"
                f"Exit: {round(stop, 2)} | Entry was: {trade['Entry']}\n"
                f"PnL: ${round((stop - trade['Entry']) * trade['Size'], 0):,.0f}"
            )
            send_telegram(msg)
            log_trade(symbol, trade["Entry"], stop, trade["Size"])
            continue

        # TARGET HIT
        if latest['High'] >= trade["Target"]:
            msg = (
                f"🎯 TARGET HIT: {symbol}\n"
                f"Exit: {trade['Target']} | Entry was: {trade['Entry']}\n"
                f"PnL: ${round((trade['Target'] - trade['Entry']) * trade['Size'], 0):,.0f}"
            )
            send_telegram(msg)
            log_trade(symbol, trade["Entry"], trade["Target"], trade["Size"])
            continue

        trade["Stop"] = round(stop, 2)
        updated.append(trade)

    save_open_trades(pd.DataFrame(updated))

# =========================================
# MAIN AGENT LOOP
# =========================================

def run_agent():

    print(f"\n{'='*50}")
    print(f"Running at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    manage_trades()

    if not market_is_bullish():
        print("📉 Market not bullish — skipping new trade scan.")
        return

    spy_df = yf.download("^GSPC", period="1y", interval="1d", progress=False)
    if spy_df is None or spy_df.empty:
        print("⚠️ Failed to download S&P data")
        return

    spy_df = spy_df.dropna()
    spy_df.columns = spy_df.columns.get_level_values(0)

    open_trades   = load_open_trades()
    open_symbols  = open_trades["Symbol"].tolist()

    if len(open_trades) >= MAX_OPEN_TRADES:
        print(f"📊 Max open trades ({MAX_OPEN_TRADES}) reached — no new entries.")
        return

    # NEW: count currently open trades per sector for concentration check
    open_sector_counts = {}
    if "Sector" in open_trades.columns:
        open_sector_counts = open_trades["Sector"].value_counts().to_dict()

    new_trades = []

    for stock in stocks:
        time.sleep(1)

        if stock in open_symbols:
            continue

        # Sector ETF strength filter
        if stock in sector_map:
            etf = sector_map[stock]
            if not sector_is_strong(etf):
                print(f"  {stock}: sector {etf} not strong — skip")
                continue

        # NEW: earnings date filter
        if is_near_earnings(stock):
            continue

        result = check_stock(stock, spy_df)

        if result:
            # NEW: sector concentration check
            sector = result.get("Sector", "OTHER")
            current_count = open_sector_counts.get(sector, 0)
            # also count how many new_trades already picked from this sector
            new_sector_count = sum(1 for t in new_trades if t.get("Sector") == sector)
            if (current_count + new_sector_count) >= MAX_SAME_SECTOR:
                print(f"  {stock}: sector {sector} already at max ({MAX_SAME_SECTOR}) — skip")
                continue

            print(f"  ✅ {stock} qualifies — score {result['Score']}")
            new_trades.append(result)

    if new_trades:

        df_new = pd.DataFrame(new_trades)

        # Rank by score first, then by reward potential
        df_new["Reward"] = df_new["Target"] - df_new["Entry"]
        df_new = df_new.sort_values(by=["Score", "Reward"], ascending=False)

        # Keep only best trades up to remaining slot count
        slots     = MAX_OPEN_TRADES - len(open_trades)
        df_new    = df_new.head(min(3, slots))

        for _, trade in df_new.iterrows():
            send_telegram(
                f"🚀 NEW SWING TRADE\n\n"
                f"Symbol : {trade['Symbol']}\n"
                f"Theme  : {trade['Sector']}\n"
                f"Score  : {trade['Score']}/28\n"
                f"Entry  : {trade['Entry']}\n"
                f"Stop   : {trade['Stop']}\n"
                f"Target : {trade['Target']}\n"
                f"Size   : {trade['Size']} shares\n"
                f"Risk   : ${round((trade['Entry'] - trade['Stop']) * trade['Size'], 0):,.0f}\n"
                f"Reward : ${round((trade['Target'] - trade['Entry']) * trade['Size'], 0):,.0f}"
            )

        combined = pd.concat([open_trades, df_new], ignore_index=True)
        save_open_trades(combined)

    else:
        print("🔍 No qualifying trades found this scan.")

    print(f"✅ Scan complete — {len(open_trades)} open trades.")

# =========================================
# RUN
# =========================================

if __name__ == "__main__":
    print("🚀 Swing Trading Agent Started")
    run_agent()
    print("✅ Done")
