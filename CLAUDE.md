# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working rules (from the owner)

- **Always log the problem-solving process** in `docs/problem_log.md`: problem, evidence, decision, outcome. Newest entry last.
- **Major problems only, major solutions only.** Identify the one problem that actually blocks or invalidates the work, and fix that. No cosmetic cleanups, speculative hardening, or side refactors. Say what was skipped.
- Findings must be measured, not claimed: run the check (backtest, IC test, test suite) before reporting a result.

## Commands

All Python runs through poetry; bare `python` is not on PATH. Imports assume `src/` is the root (`sys.path`/`PYTHONPATH=src`).

```
poetry install
poetry run python src/main.py              # one US+KR analysis pass (paper trading is the hard default)
poetry run python src/main.py --daemon     # 24/7 APScheduler (data 30m, analysis 60m, reports)
poetry run python src/main.py --us-only | --kr-only | --web [--port N]
./run_stock_ml.sh                       # FastAPI + SPA on :8000 (kills nothing; fails if port taken)
./run_backtest_gui.sh                      # Streamlit backtest dashboard
./run_backtest.sh [--start --end --top-n --rebalance --benchmark --db PATH --no-html]
cd src && PYTHONPATH=. poetry run pytest stock_ml/tests            # all (currently all skipped)
cd src && PYTHONPATH=. poetry run pytest stock_ml/tests/test_platform.py -k NAME   # single test
```

`--live` sends real Kiwoom orders. Never pass it unless explicitly told.

## Architecture

**Live pipeline** (`src/main.py`): `Investor` -> `DecisionMaker` (US) / `KRDecisionMaker` (KR) -> analysts in `src/analysts/` (macro, fundamental, stock_data, news_media, business, KR stock). Regime comes from `regime_detector/{us,kr}_regime.py`, timing from `market_timer/`, execution through `brokers/` (Kiwoom, paper by default), reporting through `reporters/` (Telegram/Discord). `scheduler/scheduler.py` runs these as recurring jobs.

**Data**: DuckDB at `src/db/stockml.duckdb` (tables `price_daily_full`, `fundamentals_q_full`, `news_sentiment`, `macro_index_full`); CSV copies in `data/raw/`. Tickers/benchmarks live in `src/config/config.yaml` (`tickers`, `benchmark_tickers` = SPY/QQQ, price-only, never pickable).

**Backtest** (`src/backtest/`): `engine.py` replays monthly/weekly rebalances from DuckDB with `date < rebalance_date` queries; `portfolio.py` does cash/slippage/commission accounting; `performance.py`/`reporter.py` produce stats. `stock_ml/backend/` wraps it in FastAPI and asks Claude or Gemini (models starting `gemini`, key `Gemini_API_KEY` in `.env`) to propose strategies.

**Research** (`src/research/`, not part of the live path): `backfill_prices.py` (2010+, adjusted) and `backfill_pool.py` (S&P 500 training pool) write `data/research/research.duckdb` (gitignored, rebuildable); `factor_ic.py` is the rank-IC baseline harness. Pass that DB to the backtest with `--db`.

## Gotchas

- The live daemon holds the write lock on `stockml.duckdb` and never checkpoints, so recent data sits in the `.wal`. Never open it read-write; copy the `.duckdb` **and** `.wal` (as `BacktestEngine` does) or use the research DB.
- Scoring logic is duplicated in `backtest/engine.py::_score_stocks` and `core/decision_maker.py::_score_stocks`. Change both or neither.
- The live DB has ~1 year of prices; any backtest on it is statistically meaningless (250-day Sharpe SE ~1). Use the research DB for evaluation.
- Backtest results on the 56-ticker universe are inflated by hindsight (winners picked today). Treat CAGR as an upper bound; judge signals by rank-IC, not backtest return.
- Known scoring flaws (see `docs/implementation_plan.md`): volatility term negligible, no skip-month, revenue-CV rewarded, no cross-sectional normalization, fundamentals ignore the ~60-day filing lag.

## Established research results

12-1 momentum rank-IC is 0.042 (t 2.35) on 56 stocks 2011-2026. Ridge and gradient boosting, including models trained on a 448-stock pool, did **not** beat it out of sample (pool-trained were significantly worse). No ML goes into the portfolio unless it beats this baseline in a purged walk-forward test. Details and the full roadmap: `docs/implementation_plan.md`.
