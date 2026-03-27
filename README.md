# Swing Trading Agent — Deployment Guide

## Files in this repo
- `swing_trading_agent.py` — main trading agent
- `requirements.txt` — Python dependencies
- `.github/workflows/run_agent.yml` — runs the agent every 30 minutes via GitHub Actions

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
