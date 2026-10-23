# Coinone RSS Buzz Bot

Monitors crypto RSS feeds + NewsAPI. When a coin gets **famous enough** (buzz threshold), fires a BUY signal with a calculated win ratio and position size.

## How It Works

1. **Scans** ~4 RSS feeds (CoinDesk, CoinTelegraph, CryptoNews, Decrypt) every 5 minutes
2. **Counts mentions** of each supported coin across all articles
3. **Scores sentiment** — bullish vs bearish keyword matching
4. If `mentions ≥ 5` AND `sentiment > 0` → **BUY signal**
5. Calculates:
   - **Win Ratio %** — heuristic estimate based on buzz intensity (55%–80%)
   - **Position %** — suggested position size (2%–10% of portfolio)
6. Sends Discord embed notification
7. (Optional) Forwards to your `server_torico /api/signal` for live execution

## Win Ratio Model

| Buzz Score | Base Win Ratio | Sentiment Bonus |
|-----------|---------------|-----------------|
| 5–9       | 55–59%        | +1.5% per point (max +8%) |
| 10–19     | 60–69%        | +1.5% per point (max +8%) |
| 20+       | 72%+          | +1.5% per point (max +8%) |
| Hard cap  | 80%           | — |

⚠️ These are **heuristic estimates**, not financial guarantees.

## Setup

```bash
cd coinone-rss-bot
npm install
npm run dry     # safe dry run (no orders placed)
npm run live    # LIVE mode (places real orders via server_torico)
```

## Environment Variables (from workspace .env)

| Variable | Source |
|----------|--------|
| `COINONE_ACCESS_KEY` | ✅ Already in workspace .env |
| `COINONE_SECRET` | ✅ Already in workspace .env |
| `NEWS_API_KEY` | ✅ Already in workspace .env |
| `DISCORD_WEBHOOK` | Add to workspace .env if you want Discord alerts |
| `SIGNAL_SERVER` | Default: `http://localhost:4000` (server_torico) |
| `SIGNAL_API_KEY` | Optional: for server_torico auth |
| `DRY_RUN` | `true` (safe) or `false` (live) |

## Supported Coins

BTC, ETH, XRP, SOL, DOGE, ADA, AVAX, MATIC, LINK, DOT

## Cooldown

30 minutes per coin after a signal fires — prevents rapid duplicate signals on the same hype wave.
