# Problem-solving log

Format: problem -> evidence -> decision -> outcome. Major problems only.

## 2026-10-28 — Backtest could not be trusted
- **Problem:** price DB covered ~1 year (2024-11 to 2025-11); a 250-day Sharpe (0.49) has standard error ~1, so no strategy or ML result could be judged.
- **Evidence:** `reports/data_scale.md`; 1-year backtest underperformed SPY (13.3% vs 21.9%).
- **Decision:** backfill adjusted prices from 2010 into a separate research DB (live DB is write-locked by the daemon), add `--db` to the backtester.
- **Outcome:** 229,921 rows, 58 tickers; 2012-2025 backtest CAGR 29.9% vs SPY 14.9%, but flagged as hindsight-inflated (hand-picked winners).

## 2026-10-28 — Is ML worth building?
- **Problem:** unclear whether any ML signal beats simple momentum; LSTM next-day prediction was already at 50.3% directional accuracy.
- **Evidence:** rank-IC harness. 12-1 momentum IC 0.042 (t 2.35); ridge 0.030; pool-trained (448 S&P 500 names, ~110k rows) ridge/HGB -0.02 to 0.00, significantly below baseline (paired t -1.7 to -2.4); dropping pool winners changed nothing.
- **Decision:** no ML in the portfolio; keep rule-based momentum, fix scoring bugs, add holding buffer and volatility target.
- **Outcome:** recorded in `docs/implementation_plan.md`; revisit only with point-in-time, delisted-inclusive data and richer features.

## 2026-10-28 — Scoring fix, 12h reports, Telegram, Plunge Detector, rename
- **Problem:** (1) live score used 60-day momentum (rank-IC 0.013, t 0.70) plus wrong-sign revenue-CV, duplicated in engine and decision maker; (2) reports every 3h; (3) Telegram sends failed.
- **Evidence:** `factor_ic` harness, 189 monthly dates, 56 stocks: old live-style 0.013, 12-1 momentum 0.042 (t 2.35), 12-1 minus 0.25/0.5 z(vol) 0.039/0.035 (vol adds nothing). Telegram: bot token and chat ID valid, HTTP 400 "can't parse entities" because Markdown mode rejects `_`/`*` in report text. Backtest 2012-2025 research DB: new CAGR 27.2% / Sharpe 0.93 vs old 29.9% / 1.10 — within noise (old engine formula was already ~12m momentum, IC 0.041); backtest return is not the criterion.
- **Decision:** one shared `core/scoring.py` (12-1 momentum z-score) used by engine, StockDataAnalyst and DecisionMaker; dropped vol/drawdown/fundamental/news terms (no measured edge or no PIT data). Telegram retries plain text on parse error. `send_report` 3h -> 12h. Plunge Detector (only config + Node receiver survived; poller lost) rebuilt as optional `src/plunge/` + `/api/plunge/*` + `PLUNGE_ENABLED=1` job. `ai_platform` -> `stock_ml`.
- **Outcome:** tests 9 + 7 pass; Telegram and Discord test messages delivered. Not done: KR decision maker still on old formula; holding buffer not built; running daemon (pid at time of work) needs restart to pick up changes.
