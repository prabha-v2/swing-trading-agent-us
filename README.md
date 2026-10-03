# Swing Trading Agent — Deployment Guide

## Files in this repo
- `swing_trading_agent_us.py` — main trading agent
- `backtest.py` — backtesting module
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

## Tracking your trades (positions.csv)
When you buy a stock from an alert, add a row to `positions.csv` (edit it on GitHub):

```
symbol,shares,entry_price,sector
NVDA,2.5,224.55,SMH
```

Use the sector shown in the alert. Fractional shares are fine. Delete the row after you sell.

Every run checks each held stock and sends one Telegram SELL alert when:
- 🛑 the stop is hit,
- 🎯 the target is hit, or
- ⏰ it has been held 30 days without hitting either.

Stop, target and buy date are taken from the stock's alert in `trade_log.csv` (if its entry
price is within 5% of yours). To set your own, add optional `stop`, `target` and `date`
(YYYY-MM-DD) columns to the header and fill them in; blank cells use the defaults.
`sell_alerts.csv` is written by the agent to avoid repeating alerts — don't edit it.

Held stocks are also skipped as new buy alerts and count toward the 60% total / 20% per
sector limits (based on `ACCOUNT_SIZE` in the agent).
