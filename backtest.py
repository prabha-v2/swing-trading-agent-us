"""
backtest.py — Backtest of the Momentum Dip strategy used by swing_trading_agent_us.py.

Rules (same constants as the live agent):
  - Universe : S&P 500 + curated list (the agent's get_dynamic_universe), price >= $5,
               20-day avg dollar volume >= $2M
  - Leader   : 6-month momentum (close 21 bars ago vs 126 bars ago) in the top
               MOM_TOP_PCT of the universe that day
  - Uptrend  : close > 200-day SMA
  - Dip      : 2-day RSI < DIP_RSI2_MAX
  - Picks    : up to TOP_PICKS per day, strongest momentum first, max MAX_PER_SECTOR
               per sector, no new signal on a symbol that still has one open
  - Entry    : signal-day close (or next open with --next-open)
  - Exit     : stop at entry - DIP_STOP_ATR x ATR(14) (gaps fill at the open);
               otherwise first close above the 5-day SMA, or the close of day DIP_MAX_HOLD

It also reports a RANDOM baseline (random liquid stocks held for the same number of
days) — a strategy is only worth running if it clearly beats that.

Caveats: the universe is today's index members (survivorship bias flatters both the
strategy and the baseline); the earnings filter isn't simulated; no commissions or
slippage are deducted (subtract ~0.1% per trade).

Usage:
    python backtest.py                    # full universe since 2018
    python backtest.py --start 2022-01-01
    python backtest.py --next-open        # enter at next day's open instead of the close
"""

import argparse
import numpy as np
import pandas as pd
import ta

import swing_trading_agent_us as agent


def compute(df):
    """Per-symbol indicator frame (no look-ahead: every column uses data up to that bar)."""
    c, h, l = df['Close'], df['High'], df['Low']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return pd.DataFrame({
        "open":   df['Open'],
        "high":   h,
        "low":    l,
        "close":  c,
        "sma5":   c.rolling(5).mean(),
        "sma200": c.rolling(200).mean(),
        "rsi2":   ta.momentum.rsi(c, window=2),
        "atr":    tr.rolling(14).mean(),
        "mom":    c.shift(21) / c.shift(126) - 1,
        "dv":     (c * df['Volume']).rolling(20).mean(),
    })


def simulate(f, i, next_open):
    """Trade signalled at bar i. Returns (pct_return, bars_held, outcome) or None if it runs past the data."""
    n = len(f)
    if next_open:
        if i + 1 >= n:
            return None
        entry, first = f.open.iat[i + 1], i + 1
    else:
        entry, first = f.close.iat[i], i + 1
    stop = entry - agent.DIP_STOP_ATR * f.atr.iat[i]
    for k, j in enumerate(range(first, min(first + agent.DIP_MAX_HOLD, n)), 1):
        if f.low.iat[j] <= stop:
            return min(f.open.iat[j], stop) / entry - 1, k, "STOPPED"
        if f.close.iat[j] > f.sma5.iat[j] or k == agent.DIP_MAX_HOLD:
            return f.close.iat[j] / entry - 1, k, "EXITED"
    return None


def run(start, next_open, seed=0):
    umap = agent.get_dynamic_universe(agent.sector_map)
    syms = sorted(umap)
    print(f"Downloading {len(syms)} symbols...")
    frames = agent.batch_download(syms, period="max", interval="1d")
    feats = {s: compute(df.dropna()) for s, df in frames.items() if len(df.dropna()) > 260}

    # Daily cross-sectional momentum rank among tradable names
    panel = pd.DataFrame({s: f.mom.where((f.close >= 5) & (f.dv >= 2e6)) for s, f in feats.items()})
    rank = panel.rank(axis=1, pct=True)
    dates = rank.index[rank.index >= start]

    rng = np.random.default_rng(seed)
    open_until, rand_open_until = {}, {}
    trades, rand = [], []
    for d in dates:
        r = rank.loc[d].dropna()
        leaders = r[r >= 1 - agent.MOM_TOP_PCT].index
        hits = []
        for s in leaders:
            f = feats[s]
            i = f.index.get_loc(d)
            if f.close.iat[i] > f.sma200.iat[i] and f.rsi2.iat[i] < agent.DIP_RSI2_MAX:
                hits.append((f.mom.iat[i], s, i))
        hits.sort(reverse=True)
        per_sector, taken = {}, 0
        for _, s, i in hits:
            if taken >= agent.TOP_PICKS:
                break
            sec = umap.get(s, "OTHER")
            if per_sector.get(sec, 0) >= agent.MAX_PER_SECTOR or open_until.get(s, pd.Timestamp.min) >= d:
                continue
            res = simulate(feats[s], i, next_open)
            if res is None:
                continue
            per_sector[sec] = per_sector.get(sec, 0) + 1
            taken += 1
            open_until[s] = feats[s].index[min(i + res[1], len(feats[s]) - 1)]
            trades.append({"date": d, "symbol": s, "sector": sec, "ret": res[0], "bars": res[1], "outcome": res[2]})

            # Random baseline: a random tradable stock held for the same number of bars
            pool = [x for x in r.index if rand_open_until.get(x, pd.Timestamp.min) < d]
            x = pool[rng.integers(len(pool))]
            fx = feats[x]
            j = fx.index.get_loc(d)
            k = min(j + res[1], len(fx) - 1)
            base = fx.open.iat[j + 1] if next_open and j + 1 < len(fx) else fx.close.iat[j]
            rand.append({"date": d, "ret": fx.close.iat[k] / base - 1})
            rand_open_until[x] = fx.index[k]
    return pd.DataFrame(trades), pd.DataFrame(rand)


def report(t, rnd):
    if t.empty:
        print("No trades.")
        return
    yr = t.date.dt.year
    print(f"\n{'='*60}\nMOMENTUM DIP — {len(t)} trades, {t.date.min():%Y-%m-%d} → {t.date.max():%Y-%m-%d}\n{'='*60}")
    print(f"Avg return/trade : {t.ret.mean()*100:+.2f}%   (random baseline, same hold: {rnd.ret.mean()*100:+.2f}%)")
    print(f"Win rate         : {(t.ret > 0).mean()*100:.0f}%")
    print(f"Avg hold         : {t.bars.mean():.1f} days")
    print(f"Stopped out      : {(t.outcome == 'STOPPED').mean()*100:.0f}%")
    print(f"Worst / 5th pct  : {t.ret.min()*100:+.1f}% / {t.ret.quantile(.05)*100:+.1f}%")
    by = pd.DataFrame({
        "trades":   t.groupby(yr).size(),
        "avg %":    t.groupby(yr).ret.mean() * 100,
        "win %":    t.groupby(yr).ret.apply(lambda r: (r > 0).mean() * 100),
        "random %": rnd.groupby(rnd.date.dt.year).ret.mean() * 100,
    })
    print("\nBy year:\n" + by.round(2).to_string())
    print("\nBy sector:\n" + t.groupby("sector").ret.agg(["count", "mean"]).assign(mean=lambda x: x["mean"] * 100)
          .rename(columns={"mean": "avg %"}).round(2).sort_values("count", ascending=False).to_string())
    out = f"backtest_results_{pd.Timestamp.now():%Y%m%d_%H%M}.csv"
    t.to_csv(out, index=False)
    print(f"\n📊 Trades saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Momentum Dip backtest")
    ap.add_argument("--start", default="2018-06-01", help="first signal date (default 2018-06-01)")
    ap.add_argument("--next-open", action="store_true", help="enter at the next day's open instead of the signal close")
    args = ap.parse_args()
    report(*run(pd.Timestamp(args.start), args.next_open))
