# Swing Trading Agent — Deployment Guide

## Strategy — Momentum Dip
The agent buys short, sharp pullbacks in the market's strongest stocks:

- **Leader**: 6-month momentum in the top 30% of the scanned universe
- **Uptrend**: close above the 200-day SMA
- **Dip**: 2-day RSI below 10
- **Stop**: entry − 2.5 × ATR(14)
- **Exit**: first close above the 5-day SMA, or after 10 trading days. The agent sends a Telegram **SELL** alert when an exit fires.

Up to 5 new picks per scan, strongest momentum first, max 2 per sector. Stocks with earnings within 12 days are skipped. Claude adds a conviction note and cautions to each pick but doesn't remove any.

Backtest (2018–2026, `python backtest.py`): about +0.58% per trade over about 3.4 days, 68% winners, ahead of a random-stock baseline in 8 of 9 years. The universe is today's index members, so results are somewhat optimistic; subtract about 0.1% per trade for costs.

The previous multi-indicator score was retired after testing showed its picks did no better than random stocks.

## Files in this repo
- `swing_trading_agent_us.py` — main trading agent
- `backtest.py` — Momentum Dip backtest with a random-stock baseline
- `requirements.txt` — Python dependencies
- `.github/workflows/run_agent.yml` — runs the agent on a schedule via GitHub Actions (5 fixed times/day during US market hours — see the cron entries in the workflow file; each run scans once and exits)

## Setup Steps (do this once)

### Step 1 — Add your Telegram secrets in GitHub
1. Go to your repo on GitHub
2. Click Settings → Secrets and variables → Actions → New repository secret
3. Add: `TELEGRAM_TOKEN` = your bot token
4. Add: `CHAT_ID` = your chat ID

### Step 2 — Enable GitHub Actions
1. Click the Actions tab in your repo
2. Click "I understand my workflows, go ahead and enable them"

That's it. The agent runs every 30 minutes automatically.

## Manually trigger a run
Go to Actions tab → "Swing Trading Agent" → Run workflow → Run workflow

## Check logs
Go to Actions tab → click any run → click "run-agent" job to see full output

## Important notes
- GitHub Actions is FREE for public repos (unlimited minutes)
- For private repos: free tier gives 2,000 minutes/month
  - Each run takes ~3-5 minutes, so 30-min schedule = ~48 runs/day = ~240 min/day
  - That's ~7,200 min/month — EXCEEDS free private repo limit
  - Solution: make the repo PUBLIC (your code is visible but secrets are protected)
  - Or: reduce frequency to every 1 hour to stay within free limits
