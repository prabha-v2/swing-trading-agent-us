"""
backtest.py — Walk-forward backtest for the swing trading scoring system.

For each stock in the universe, this script:
  1. Downloads 2 years of daily OHLCV data
  2. Computes all indicators rolling (no look-ahead)
  3. On each day where score >= threshold, simulates a trade:
       - Entry = next day's open
       - Exit = first day within MAX_HOLD_DAYS where High >= Target OR Low <= Stop
       - If neither hit, exit at close of bar MAX_HOLD_DAYS
  4. Reports: win rate, avg R-multiple, profit factor, best/worst setups

Usage:
    python backtest.py                          # full universe, 2 years
    python backtest.py --symbols NVDA AMD MSFT  # specific symbols
    python backtest.py --days 365               # 1-year lookback
    python backtest.py --threshold 20           # lower score threshold
"""

import argparse
import yfinance as yf
import pandas as pd
import ta
import time
from datetime import datetime, timedelta


# =========================================
# CONFIG
# =========================================

SCORE_THRESHOLD  = 22
RR_RATIO         = 2.5
MAX_ATR_STOP     = 3.0
MAX_HOLD_DAYS    = 20     # exit after 20 bars if neither stop nor target hit
MIN_BARS         = 300    # need enough history to compute all indicators


# =========================================
# INDICATOR COMPUTATION
# =========================================

def compute_indicators(df):
    """Add all indicator columns to df in place."""
    df = df.copy()
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

    df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']

    df['OBV']       = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['OBV_EMA20'] = ta.trend.ema_indicator(df['OBV'], window=20)

    df['MACD']      = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
    df['MACD_SIG']  = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_HIST'] = ta.trend.macd_diff(df['Close'], window_slow=26, window_fast=12, window_sign=9)

    return df


# =========================================
# SCORING FUNCTION (mirrors main agent)
# =========================================

def score_bar(df, idx, spy_slice):
    """
    Compute signal score at a specific bar index.
    Returns (score, setup_type) or (None, None) if data insufficient.
    Uses only data up to and including idx (no look-ahead).
    """
    if idx < MIN_BARS:
        return None, None

    latest = df.iloc[idx]
    prev   = df.iloc[idx - 1]

    # Basic NaN guard
    for col in ['EMA10','EMA20','EMA50','EMA200','RSI','ATR','ADX','MACD','MACD_SIG']:
        if pd.isna(latest.get(col, float('nan'))):
            return None, None

    price  = float(latest['Close'])
    if price < 5.0:
        return None, None

    avg_dv = float(df['Close'].iloc[idx-20:idx].mean() * df['Volume'].iloc[idx-20:idx].mean())
    if avg_dv < 2_000_000:
        return None, None

    score = 0

    # RS vs SPY (3/6/12 month)
    if len(spy_slice) >= 63:
        s3  = float(df['Close'].iloc[idx] / df['Close'].iloc[max(idx-63, 0)] - 1)
        s6  = float(df['Close'].iloc[idx] / df['Close'].iloc[max(idx-126,0)] - 1)
        s12 = float(df['Close'].iloc[idx] / df['Close'].iloc[max(idx-252,0)] - 1)
        n3  = float(spy_slice['Close'].iloc[-1] / spy_slice['Close'].iloc[max(-63,-len(spy_slice))] - 1)
        n6  = float(spy_slice['Close'].iloc[-1] / spy_slice['Close'].iloc[max(-126,-len(spy_slice))] - 1)
        n12 = float(spy_slice['Close'].iloc[-1] / spy_slice['Close'].iloc[max(-252,-len(spy_slice))] - 1)
        rs  = sum([s3 > n3, s6 > n6, s12 > n12])
        if rs >= 2: score += 2
        if rs == 3: score += 1

    # EMA stack
    if latest['EMA10']  > latest['EMA20']:  score += 2
    if latest['EMA20']  > latest['EMA50']:  score += 2
    if latest['EMA50']  > latest['EMA200']: score += 2
    if latest['Close']  > latest['EMA50']:  score += 1
    if latest['Close']  > latest['EMA200']: score += 2

    # RSI
    rsi      = float(latest['RSI'])
    rsi_prev = float(prev['RSI']) if not pd.isna(prev['RSI']) else rsi
    if rsi > 60 and rsi_prev < 60:         score += 2
    elif 55 < rsi < 75:                     score += 2
    elif 40 < rsi < 55 and rsi > rsi_prev: score += 1
    elif rsi > 80:                          score -= 2
    elif rsi < 40:                          score -= 1

    # Breakout
    hh20 = float(df['HH20'].iloc[idx - 1]) if not pd.isna(df['HH20'].iloc[idx-1]) else 0
    if latest['Close'] > hh20:              score += 2
    h52 = float(df['High52'].iloc[idx]) if not pd.isna(df['High52'].iloc[idx]) else latest['Close']
    ath_dist = latest['Close'] / h52 if h52 > 0 else 1.0
    if ath_dist > 0.90:  score += 2
    elif ath_dist > 0.80: score += 1

    # Volume
    avg_vol = float(df['AvgVol'].iloc[idx]) if not pd.isna(df['AvgVol'].iloc[idx]) else 1
    rvol = float(latest['Volume']) / avg_vol if avg_vol > 0 else 1.0
    if rvol > 2.0:   score += 2
    elif rvol > 1.5: score += 1

    # ATR expanding
    atr_now  = float(latest['ATR'])
    atr_5ago = float(df['ATR'].iloc[idx-5]) if idx >= 5 and not pd.isna(df['ATR'].iloc[idx-5]) else atr_now
    if atr_now > atr_5ago: score += 1

    # Entry quality
    dist = (latest['Close'] - latest['EMA20']) / latest['EMA20'] if latest['EMA20'] > 0 else 0
    if dist < 0.05:   score += 2
    elif dist < 0.08: score += 1
    elif dist > 0.20: score -= 1

    # 60-day outperformance vs SPY
    if idx >= 60 and len(spy_slice) >= 60:
        sr = float(df['Close'].iloc[idx] / df['Close'].iloc[idx-60] - 1)
        nr = float(spy_slice['Close'].iloc[-1] / spy_slice['Close'].iloc[-60] - 1)
        if sr > nr * 1.5:  score += 2
        elif sr > nr:      score += 1

    # Distribution penalty
    rh10 = float(df['High'].iloc[idx-10:idx].max())
    if latest['Close'] < rh10 * 0.85: score -= 2

    # OBV
    obv_rising   = float(df['OBV'].iloc[idx]) > float(df['OBV_EMA20'].iloc[idx])
    obv_slope    = float(df['OBV'].iloc[idx]) - float(df['OBV'].iloc[max(idx-10,0)])
    obv_trend_p  = obv_slope > 0
    if obv_rising and obv_trend_p:         score += 2
    elif obv_rising:                        score += 1
    elif not obv_trend_p and not obv_rising: score -= 2

    # MACD
    macd_now  = float(latest['MACD'])
    macd_sig  = float(latest['MACD_SIG'])
    macd_prev_v = float(prev['MACD']) if not pd.isna(prev['MACD']) else macd_now
    macd_sig_p  = float(prev['MACD_SIG']) if not pd.isna(prev['MACD_SIG']) else macd_sig
    macd_hist_n = float(latest['MACD_HIST'])
    macd_hist_p = float(prev['MACD_HIST']) if not pd.isna(prev['MACD_HIST']) else macd_hist_n

    crossed_up  = macd_prev_v < macd_sig_p and macd_now > macd_sig
    above_sig   = macd_now > macd_sig
    hist_rising = macd_hist_n > macd_hist_p and macd_hist_n > 0

    if crossed_up:                         score += 2
    elif above_sig and hist_rising:        score += 2
    elif above_sig:                        score += 1
    elif not above_sig and macd_hist_n < 0: score -= 1

    # ADX
    adx = float(latest['ADX']) if not pd.isna(latest['ADX']) else 20.0
    if adx > 30:   score += 2
    elif adx > 20: score += 1
    elif adx < 15: score -= 2

    # BB Squeeze
    bb_squeeze = False
    if idx >= 60:
        pct20 = float(df['BB_width'].iloc[idx-60:idx].quantile(0.20))
        bb_squeeze = float(df['BB_width'].iloc[idx]) < pct20
    if bb_squeeze: score += 3

    # Determine setup type
    if bb_squeeze and latest['Close'] > hh20:
        setup = "Squeeze Breakout"
    elif latest['Close'] > hh20 and rvol > 2.0:
        setup = "Volume Breakout"
    elif dist < 0.05 and rsi > 50:
        setup = "EMA20 Pullback"
    elif ath_dist > 0.95:
        setup = "ATH Breakout"
    elif bb_squeeze:
        setup = "Squeeze Setup"
    else:
        setup = "Trend Continuation"

    return score, setup


# =========================================
# TRADE SIMULATION
# =========================================

def simulate_trade(df, signal_idx, rr_ratio=RR_RATIO):
    """
    Simulate a trade triggered at signal_idx.
    Entry = next bar's open (no look-ahead on the signal bar itself).
    Stop  = entry - MAX_ATR_STOP * ATR
    Target = entry + risk * RR_RATIO
    Returns a result dict or None.
    """
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None

    entry = float(df['Open'].iloc[entry_idx])
    atr   = float(df['ATR'].iloc[signal_idx])
    if pd.isna(atr) or atr <= 0:
        return None

    # Same stop logic as live agent
    ten_bar_low = float(df['Low'].iloc[max(signal_idx-10, 0):signal_idx+1].min())
    atr_stop    = entry - (MAX_ATR_STOP * atr)
    stop        = max(ten_bar_low, atr_stop)
    risk        = entry - stop

    if risk <= 0 or risk > entry * 0.12:
        return None

    target = entry + (risk * rr_ratio)

    # Walk forward bar by bar
    outcome     = "TIMEOUT"
    exit_price  = float(df['Close'].iloc[min(entry_idx + MAX_HOLD_DAYS - 1, len(df)-1)])
    exit_bar    = MAX_HOLD_DAYS

    for i in range(1, MAX_HOLD_DAYS + 1):
        bar_idx = entry_idx + i
        if bar_idx >= len(df):
            break
        hi = float(df['High'].iloc[bar_idx])
        lo = float(df['Low'].iloc[bar_idx])

        # Both in same bar: conservative — assume stop hit first
        if lo <= stop:
            outcome    = "STOPPED"
            exit_price = stop
            exit_bar   = i
            break
        if hi >= target:
            outcome    = "TARGET"
            exit_price = target
            exit_bar   = i
            break

    r_multiple = (exit_price - entry) / risk if risk > 0 else 0

    return {
        "entry":       round(entry, 2),
        "stop":        round(stop, 2),
        "target":      round(target, 2),
        "exit_price":  round(exit_price, 2),
        "outcome":     outcome,
        "r_multiple":  round(r_multiple, 2),
        "hold_bars":   exit_bar,
        "risk_pct":    round(risk / entry * 100, 2),
    }


# =========================================
# BACKTEST ONE SYMBOL
# =========================================

def backtest_symbol(symbol, spy_df, lookback_days=730, threshold=SCORE_THRESHOLD):
    """
    Run a walk-forward backtest on one symbol.
    Returns list of trade dicts (empty if no signals).
    """
    start = (datetime.now() - timedelta(days=lookback_days + 100)).strftime('%Y-%m-%d')
    try:
        df = yf.download(symbol, start=start, interval='1d', progress=False)
        if df is None or df.empty:
            return []
        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if len(df) < MIN_BARS:
            return []
        df = compute_indicators(df)
    except Exception as e:
        print(f"  ⚠️ {symbol}: {e}")
        return []

    # Restrict to lookback window
    cutoff = datetime.now() - timedelta(days=lookback_days)
    df_full = df.copy()
    start_idx = 0
    for i, idx_val in enumerate(df_full.index):
        if hasattr(idx_val, 'to_pydatetime'):
            dt = idx_val.to_pydatetime().replace(tzinfo=None)
        else:
            dt = pd.Timestamp(idx_val).to_pydatetime().replace(tzinfo=None)
        if dt >= cutoff:
            start_idx = i
            break

    trades = []
    i      = start_idx
    last_entry_idx = -MAX_HOLD_DAYS  # avoid overlapping trades

    while i < len(df_full) - MAX_HOLD_DAYS:
        # Need matching SPY slice up to this date
        spy_slice = spy_df.iloc[:min(i + 1, len(spy_df))]

        score, setup = score_bar(df_full, i, spy_slice)

        if score is not None and score >= threshold and (i - last_entry_idx) >= MAX_HOLD_DAYS:
            result = simulate_trade(df_full, i)
            if result:
                date_val = df_full.index[i]
                if hasattr(date_val, 'to_pydatetime'):
                    date_str = date_val.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_val)[:10]

                result.update({
                    "symbol":    symbol,
                    "date":      date_str,
                    "score":     score,
                    "setup":     setup,
                })
                trades.append(result)
                last_entry_idx = i + 1
                i += MAX_HOLD_DAYS    # skip forward to avoid overlapping

        i += 1

    return trades


# =========================================
# AGGREGATE STATS
# =========================================

def print_stats(all_trades):
    if not all_trades:
        print("No trades found.")
        return

    df = pd.DataFrame(all_trades)

    total      = len(df)
    wins       = df[df['outcome'] == 'TARGET']
    losses     = df[df['outcome'] == 'STOPPED']
    timeouts   = df[df['outcome'] == 'TIMEOUT']
    win_rate   = len(wins) / total * 100

    avg_r_win  = wins['r_multiple'].mean() if len(wins) else 0
    avg_r_loss = losses['r_multiple'].mean() if len(losses) else 0
    avg_r_all  = df['r_multiple'].mean()

    gross_win  = wins['r_multiple'].sum() if len(wins) else 0
    gross_loss = abs(losses['r_multiple'].sum()) if len(losses) else 0
    pf         = gross_win / gross_loss if gross_loss > 0 else float('inf')

    avg_hold   = df['hold_bars'].mean()

    print(f"\n{'='*55}")
    print(f"BACKTEST RESULTS — {total} trades")
    print(f"{'='*55}")
    print(f"Win Rate      : {win_rate:.1f}%  ({len(wins)} wins / {len(losses)} stops / {len(timeouts)} timeouts)")
    print(f"Avg R (all)   : {avg_r_all:+.2f}R")
    print(f"Avg R (wins)  : {avg_r_win:+.2f}R")
    print(f"Avg R (losses): {avg_r_loss:+.2f}R")
    print(f"Profit Factor : {pf:.2f}")
    print(f"Avg Hold      : {avg_hold:.1f} bars")

    print(f"\n--- By Setup Type ---")
    for setup, grp in df.groupby('setup'):
        wr = len(grp[grp['outcome']=='TARGET']) / len(grp) * 100
        ar = grp['r_multiple'].mean()
        print(f"  {setup:22} {len(grp):3} trades | WR:{wr:.0f}% | AvgR:{ar:+.2f}")

    print(f"\n--- By Score Bucket ---")
    df['score_bucket'] = pd.cut(df['score'], bins=[0,24,27,30,99], labels=['22-24','25-27','28-30','31+'])
    for bucket, grp in df.groupby('score_bucket', observed=True):
        wr = len(grp[grp['outcome']=='TARGET']) / len(grp) * 100 if len(grp) else 0
        ar = grp['r_multiple'].mean() if len(grp) else 0
        print(f"  Score {bucket}: {len(grp):3} trades | WR:{wr:.0f}% | AvgR:{ar:+.2f}")

    print(f"\n--- Top 10 Best Trades ---")
    top = df.nlargest(10, 'r_multiple')[['symbol','date','setup','score','r_multiple','outcome','hold_bars']]
    print(top.to_string(index=False))

    print(f"\n--- Top 10 Worst Trades ---")
    worst = df.nsmallest(10, 'r_multiple')[['symbol','date','setup','score','r_multiple','outcome','hold_bars']]
    print(worst.to_string(index=False))

    # Save to CSV
    out_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(out_file, index=False)
    print(f"\n📊 Full results saved to {out_file}")


# =========================================
# MAIN
# =========================================

DEFAULT_SYMBOLS = [
    # A representative cross-section for a meaningful backtest
    "NVDA","AMD","AVGO","MSFT","AAPL","META","GOOGL","AMZN","TSLA",
    "PLTR","CRWD","PANW","NET","SNOW","DDOG","NOW","CRM","ADBE",
    "GS","JPM","V","MA","COIN","HOOD",
    "LMT","RTX","NOC","GD","RKLB",
    "LLY","NVO","VKTX",
    "FSLR","ENPH","CEG","VST",
    "COST","WMT","MELI",
    "XOM","CVX","MPC",
    "EQIX","DLR",
    "NVDA","MU","TSM","AMAT",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swing Trading Backtest")
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Symbols to backtest (default: built-in list of ~40)')
    parser.add_argument('--days', type=int, default=730,
                        help='Lookback period in calendar days (default: 730 = 2 years)')
    parser.add_argument('--threshold', type=int, default=SCORE_THRESHOLD,
                        help=f'Min score threshold (default: {SCORE_THRESHOLD})')
    args = parser.parse_args()

    symbols   = list(dict.fromkeys(args.symbols or DEFAULT_SYMBOLS))
    lookback  = args.days
    threshold = args.threshold

    print(f"🔬 Backtest: {len(symbols)} symbols | {lookback}d lookback | score>={threshold}")
    print(f"Started at {datetime.now().strftime('%H:%M:%S')}\n")

    # Download SPY once for RS calculations
    print("Downloading SPY reference data...")
    spy_start = (datetime.now() - timedelta(days=lookback + 300)).strftime('%Y-%m-%d')
    spy_df    = yf.download("^GSPC", start=spy_start, interval='1d', progress=False)
    spy_df    = spy_df.dropna()
    spy_df.columns = spy_df.columns.get_level_values(0)

    all_trades = []
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}", end='', flush=True)
        trades = backtest_symbol(sym, spy_df, lookback_days=lookback, threshold=threshold)
        print(f" → {len(trades)} trades")
        all_trades.extend(trades)
        time.sleep(0.5)

    print_stats(all_trades)
