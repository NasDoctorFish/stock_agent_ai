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
