# Architecture Audit — Stock_ML

Date: 2026-10-24
Scope: Stage 0 audit per the Quantitative Portfolio Research & Risk Management
Platform master prompt. Read-only investigation; no behavior changed except
two pre-existing bugs fixed in the same session this audit was written in
(see "Fixed this session" below) because they blocked verifying the system
at all.

## 1. Repository structure

```
src/
  analysts/        Signal/metric producers (one per data domain): macro
                    (LSTM-based regime prob.), fundamental, stock_data
                    (momentum/vol/drawdown), news_media, business,
                    kr_stock_analyst (KR equivalent, yfinance .KS tickers)
  backtest/         engine.py, portfolio.py, performance.py, reporter.py,
                    run_backtest.py (CLI), dashboard.py (Streamlit GUI)
  brokers/          base_broker.py (ABC), kiwoom_broker.py (KR equities),
                    coinone_broker.py (crypto), kr_order_executor.py
                    (decision -> order bridge)
  core/             db_handler.py (DuckDB I/O), db_updater.py +
                    db_updater_utils.py (CSV fetch/merge/save pipeline),
                    decision_maker.py (US), kr_decision_maker.py (KR),
                    investor.py (aggregates analysts), status_logger.py
  scheduler/        scheduler.py (APScheduler jobs, daemon entry point)
  reporters/        daily_report.py, discord_notifier.py, telegram_sender.py,
                    report_generator.py
  config/           config.yaml (ticker universe + benchmark_tickers),
                    kr_tickers.yaml, macro_indices_dict.yaml, schema_macro.py
  db/               stockml.duckdb (+ .wal — see Finding 1)
  web/              server.py (separate small FastAPI/Flask-style web server,
                    not part of the Streamlit app)
main.py             CLI entry: one-shot run, --daemon, --kr-only/--us-only,
                    --web, --paper/--live
run.sh              Launches main.py --daemon (pm2-managed, process "Stock_ML")
run_backtest.sh      Launches run_backtest.py with DB-real date defaults
run_backtest_gui.sh  Launches dashboard.py (Streamlit)
```

No `tests/` directory exists anywhere in the repo. Zero automated tests.

## 2. Environment & dependencies

Poetry-managed (`pyproject.toml`), Python >=3.10,<3.12, installed interpreter
3.11. Key libs already present: `duckdb==1.4.1`, `streamlit>=1.31,<1.52`
(1.51.0 installed), `plotly>=7.1.0,<8.0.0`, `numpy==1.26.4`, `pandas`,
`scikit-learn`, `torch`, `tensorflow` (the latter two are large and mostly
unused — only `macro_analyst.py` imports `torch`; grep found zero
`tensorflow` usage outside `pyproject.toml` and zero `sklearn` usage in the
decision path). No `scipy`/`cvxpy` currently installed — needed for Phase 1
(portfolio optimization).

## 3. Signal generation — what exists today

**Not machine learning, except one component.** Verified by grepping the
entire scoring path for `.predict(`, model loads, `sklearn`, `xgboost`: zero
hits outside `macro_analyst.py`.

- `decision_maker.py::_score_stocks` (US) and `backtest/engine.py::_score_stocks`
  (backtest's own parallel reimplementation): hand-weighted linear formula —
  `score = 2.0*momentum - 0.5*volatility + 0.1*drawdown + 1.0*revenue_growth
  + 0.5*net_margin + 0.3*(sentiment-0.5)`. Pure arithmetic, no model.
- `macro_analyst.py::MacroTrendLSTM` — a real PyTorch LSTM, trained on the
  fly (`fit`-style loop with Adam/BCELoss, ~10s per run per the session's
  observed logs) on macro index data, producing a bullish-probability score
  for the LIVE US regime classification only. Falls back to a heuristic if
  torch is unavailable. **The backtest does not use this LSTM at all** — by
  the engine's own documented design ("No live API calls"), it re-derives
  regime from 6 hand-coded heuristic signals instead
  (`engine.py::_compute_regime`).
- KR side: `kr_decision_maker.py` — structure not fully audited line-by-line
  this session; same general shape as the US decision_maker (score + regime
  -> decision dict), confirmed via `KROrderExecutor`'s consumption of its
  output shape (`economic_regime`, `equities_favorable`, `top_stocks`).

## 4. Market regime classification

Two independent, disagreeing implementations exist:
1. **Live US**: `macro_analyst.py`'s LSTM, feeding `decision_maker.py`.
2. **Backtest US**: `engine.py::_compute_regime` — 6 heuristic votes (growth,
   momentum, volatility, trend, breadth, rates) summed to a score in
   [-6, +6], mapped to Risk-On/Cautious/Defensive/Risk-Off via fixed
   thresholds (`_classify_regime`), which also sets equity allocation
   fraction (100%/70%/40%/0%) via `_equity_fraction`.

These two will **not** produce the same regime label for the same date —
the backtest is a deliberate simplification, not a bug, but it means
backtest performance numbers are not a faithful replay of what the live
LSTM-driven system would have done historically. Worth stating explicitly
in any results the platform surfaces (financial_assumptions.md captures
this).

KR regime: `KRRegimeDetector` (`regime_detector/kr_regime.py`, not read in
depth this session) — separate again.

## 5. Top-N selection / rebalancing — confirmed root cause of the "3-day
   sell" symptom

Verified empirically this session by running real backtests and inspecting
`PerformanceCalculator.get_round_trips()`'s holding-period distribution:

| Rebalance freq | Min holding (days) | Median | Sample |
|---|---|---|---|
| monthly | 28 | 61 | 96 round trips, 2024-11-06→2025-11-05 |
| weekly | 7 | 21 | 493 round trips, same window |

**Root cause**: `_score_stocks` re-ranks the entire universe from scratch
every rebalance with no persistence/momentum bonus for existing holdings and
no minimum-holding-period floor. `_build_target_weights` always reconstructs
top-N fresh each cycle (`engine.py`). Any stock that drops even slightly
below the N-th rank gets fully sold at the very next rebalance — so holding
periods cluster right at the rebalance interval (28d/7d minimums above), not
because of any single bug but because the strategy has **no hysteresis**.
The user's "3 days" observation doesn't exactly match either measured
minimum; worth re-confirming which rebalance frequency and date range
produced that specific observation before fixing, but the mechanism
(no minimum hold, no re-entry penalty, full reshuffle every cycle) is
unambiguous and is the right thing to fix regardless of the exact number.

A **second, separate** short-holding mechanism exists in the **live** KR
path only: `KROrderExecutor._execute_sell_signals` liquidates **all**
open positions unconditionally the moment `equities_favorable` flips False
— no holding-period check there either. This is not used in the backtest
(the backtest has its own equity_fraction scaling instead of a binary
liquidate-everything switch) and is not currently on any schedule (see
Finding 2) — but if the KR live path is ever scheduled, this is the second
place a holding-period floor needs to apply.

## 6. Backtesting implementation — what `./run_backtest.sh` actually supports

Verified directly against `src/backtest/run_backtest.py`'s argparse:
`--start`, `--end`, `--rebalance {weekly,monthly}`, `--top-n`,
`--initial-capital`, `--benchmark`, `--output`, `--no-html`, `--no-csv`,
`--log-level`. All confirmed working end-to-end this session (CLI, HTML,
CSV, and the Streamlit GUI). No walk-forward validation, no strategy
comparison, no multiple allocation methods (equal-weight only), no
portfolio-level risk engine beyond the stats already in
`PerformanceReport` — these are the real gaps Phase 1-3 need to fill, not
things that are broken.

No look-ahead bias found in the price-selection logic
(`_prices_on_date` filters strictly `date < rebal_date`, confirmed by
reading and by the module's own stated design principle).

## 7. Benchmark configuration — audited and fixed this session

- Default was `SPY`, which **did not exist in the DB** (56 single-name
  tickers only, no index ETF) — confirmed by direct query. Fixed by
  fetching real SPY/QQQ price history via the existing `fetch_prices()`
  utility and merging into `data/raw/price_daily.csv`.
- `benchmark_ticker` reaches the engine correctly (`BacktestEngine.__init__`
  -> `_fetch_benchmark_prices`), confirmed via `--benchmark AAPL` and
  `--benchmark SPY` producing real, non-empty `benchmark_series` and
  correct Alpha/Beta/Information Ratio output.
- Benchmark prices are aligned by `pd.Timestamp` index intersection with
  strategy returns before Beta/IR calculation (`performance.py::compute`,
  `common = strat_returns.index.intersection(bench_returns.index)`) — dates
  not present in both series are dropped, not forward-filled or
  mismatched.
- Dividends: `fetch_prices()` captures `Dividends`/`Stock Splits` columns
  from yfinance but **NAV/return calculations only ever read `close`**
  (confirmed: `engine.py`'s price queries select `close` only). So
  **benchmark (and strategy) returns are price-return, not total-return** —
  dividends are fetched but not applied anywhere. For SPY (~1.3% yield) over
  a 1-year window this understates benchmark total return by roughly that
  order of magnitude. Flagged as a financial-assumptions item, not silently
  fixed.
- **Fixed this session**: SPY/QQQ were excluded from being accidentally
  picked as live/backtest BUY candidates via a new `config.yaml::benchmark_tickers`
  list, filtered in both `decision_maker.py::_score_stocks` and
  `engine.py::_score_stocks`. Verified: neither appears in a backtest trade
  log even at wide `top_n`.
- Never silently substitutes an unavailable benchmark — when benchmark data
  is empty, `benchmark_total_return`/`beta`/`alpha`/`information_ratio` all
  surface as `nan`/"n/a", not a fallback ticker (confirmed in both console
  and GUI rendering).

## 8. Performance metric calculations — formula audit

All in `src/backtest/performance.py::PerformanceCalculator.compute()`,
read in full this session:

| Metric | Formula / method | Assessment |
|---|---|---|
| Total Return | `nav[-1]/nav[0] - 1` | Correct |
| CAGR | `(1+total_return)^(1/years) - 1`, years = calendar days/365.25 | Correct, standard |
| Annualized Volatility | `daily_returns.std() * sqrt(252)` | Correct, standard trading-day assumption |
| Max Drawdown | `(nav - nav.cummax()) / nav.cummax()`, min | Correct |
| Sharpe | `CAGR / annualized_vol` | **No risk-free rate subtracted** — this is really a Sharpe-like return/vol ratio, not textbook excess-return Sharpe. Flagged in financial_assumptions.md (master prompt rule #10/#11 territory). |
| Sortino | `CAGR / (downside_returns.std() * sqrt(252))` | Same risk-free-rate omission |
| Calmar | `CAGR / abs(max_drawdown)` | Correct, standard |
| Information Ratio | `(excess_returns.mean()*252) / (excess_returns.std()*sqrt(252))` | Correct, standard |
| Beta/Alpha | OLS via `np.cov` (`beta = cov(x,y)/var(x)`), `alpha_ann = (mean(y) - beta*mean(x)) * 252` | Mathematically correct simple OLS. This is **regression alpha**, correctly labeled as such in the UI (not conflated with simple excess return — the UI shows both `excess_return` and `alpha` as separate fields). |
| VaR/CVaR 95/99 | Historical percentile of daily returns | Correct, standard historical method (not parametric/Monte Carlo) |
| Win Rate / Avg Win / Avg Loss / Profit Factor | FIFO-matched round trips (now exposed via `get_round_trips()`, added this session) | Correct computation; see Finding below on small-sample fragility |
| Turnover | `total_gross_traded / rebalance_count / avg_nav` | Correct per-rebalance turnover proxy |

**Suspicious-looking figures investigated**: the master prompt flagged
Profit Factor ≈49, Win Rate 75%/29 trades, Alpha far exceeding cumulative
excess return as worth checking. These exact figures were reproduced this
session in ad hoc short-window test runs (e.g. `--start 2025-06-01`, ~5
months, 29 trades). Root cause confirmed: **small sample size**, not a
formula bug. Profit Factor = gross_profit/gross_loss on ~20-30 round trips
in a strongly trending short window trivially produces large ratios when
losses are few and small. CAGR/Alpha annualization of a 5-month window
naturally amplifies any absolute return by `(1/years)` where years≈0.4,
compounding small-sample noise. These are correctly-computed but
statistically unstable outputs of short evaluation windows — the fix is
methodological (minimum window length, walk-forward validation per Phase
3.7), not a formula correction.

## 9. Database structure & a serious finding (fixed this session)

`src/db/stockml.duckdb`, 4 tables: `price_daily_full`, `fundamentals_q_full`,
`macro_index_full`, `news_sentiment`. Populated from `data/raw/*.csv` via
`DBHandler.save_all_to_duckdb()` (`CREATE OR REPLACE TABLE ... AS SELECT *
FROM read_csv_auto(...)`, full replace each cycle, not incremental).

**Finding 1 (serious, fixed this session)**: the live pm2 daemon
(`src/main.py --daemon`) holds its `DBHandler`'s DuckDB connection open for
its **entire process lifetime** and never calls `CHECKPOINT` or closes/
reopens. All writes accumulate in `stockml.duckdb.wal` (observed growing to
9.5MB+ uncheckpointed) while the main `.duckdb` file's own mtime never
changes. Every piece of code that reads the DB by copying just the main
file (`BacktestEngine.__init__`, `dashboard.py::_db_meta` — both written
this session, both using the DB-copy-to-tempfile pattern to dodge the
daemon's write lock) was **silently reading stale data for as long as the
daemon stayed up**, independent of any specific data issue. This is why
SPY/QQQ didn't appear in the GUI even after being correctly merged into the
CSV and reloaded by the daemon. **Fixed**: both copy sites now also copy
the `.wal` file alongside the main file; DuckDB replays it on open. The
dashboard's cache key was also fixed (was keyed on the main file's mtime
alone, which never changes under this daemon behavior — now uses
`max(main_mtime, wal_mtime)`).

This same staleness risk applies to **anything else** that reads
`stockml.duckdb` by direct file copy while the daemon is running — worth
keeping in mind for any new data-layer code in Phase 1+.

**Finding 2**: `RUN_FETCH = False` is hardcoded in
`db_updater_utils.py::update_csv_files` (line ~106) — live price/
fundamentals fetching via `fetch_prices()` is permanently disabled. The
scheduled `job_data_update` (every 30 min) re-reads and re-saves the
**same** `data/raw/price_daily.csv` unchanged each cycle (bumping its mtime
but not its content) rather than fetching new data. `price_daily_full`'s
actual price coverage is frozen at 2024-11-06→2025-11-05 except for the
SPY/QQQ rows manually added this session (which now extend to
2026-10-23, fetched ad hoc, outside the disabled pipeline). **Not fixed
this session** — flipping `RUN_FETCH` would re-fetch all 58 tickers from
yfinance on every 30-min cycle, a much bigger operational change than
anything requested so far, and is a decision for the user, not an
audit-time fix.

**Finding 3**: The KR live order-execution path
(`KROrderExecutor`/`KiwoomBroker`) is **not wired into the scheduler at
all**. `src/scheduler/scheduler.py::job_kr_analysis` (hourly) only calls
`kr_decision_maker.run()` for scoring/regime — it never calls
`KROrderExecutor.execute()`. Order execution only happens via the one-shot
CLI path (`python main.py` without `--daemon`, see `main.py::run_kr_pipeline`
line ~212), which the currently-running pm2 daemon never invokes. So **no
paper or live trades are currently being generated automatically** by the
running system, despite the KR order-execution code existing and working.
This is the key gap behind the user's "generate actual trading logs" ask.

**Finding 4**: `KiwoomBroker`'s `paper_trading=True` mode is **not** a real
Kiwoom sandbox/test API — `place_order`/`get_balance`/`get_positions` all
short-circuit to a locally fabricated success response with zero network
calls when `paper_trading=True` (`kiwoom_broker.py` lines 234-243,
277-279, 291-293). `get_current_price`, however, does **not** short-circuit
— it still attempts a real authenticated API call even in paper mode
(lines 317-336), which will fail without real Kiwoom credentials. This is
an inconsistency worth fixing if the GUI is going to call `get_current_price`
for position sizing in paper mode. There is no evidence anywhere in the
codebase of an actual Kiwoom demo/sandbox endpoint being used — "paper
trading" here means "simulated locally," not "real test API."

## 10. Existing Streamlit components

- `src/app.py` — "Macro Dashboard," single-page, triggers a live data
  update (`DBUpdater.update_csv_files()` + `db.save_all_to_duckdb()`) on
  every load, then shows one macro feature chart via `st.pyplot`. No
  multipage structure, no `pages/` directory convention in actual use
  (only a throwaway unrelated demo at repo-root `streamlit_test/`).
- `src/backtest/dashboard.py` — built this session. Single page: sidebar
  params (DB-driven date range & ticker list, not hardcoded), run button,
  cached execution (`st.cache_data`), metrics cards, equity/drawdown chart
  (shared `BacktestReporter.build_figure`), formatted/styled stats table,
  regime-attribution bar chart, filterable Trade Log, Investment Records
  (round-trip P&L) with CSV exports. Validated against a colorblind-safe
  reference palette (dataviz skill). Verified via Streamlit's `AppTest`
  headless harness — zero exceptions, correct element counts.
- `src/web/server.py` — a separate, smaller web server (likely
  FastAPI/Flask-style, not read in depth this session) serving a crypto
  ticker JSON endpoint and a KR decision endpoint. Unrelated to Streamlit.

## 11. Report & CSV generation

`BacktestReporter` (`backtest/reporter.py`): console (ANSI-colored,
tabulate-or-fallback), CSV (NAV+benchmark, flattened stats dict), HTML
(self-contained Plotly report via CDN). All three confirmed working this
session, including after two internal refactors (`build_figure` extraction,
`stats_rows` extraction) done to let the GUI share the same formatting/
charting code rather than duplicate it.

Separately, `reporters/daily_report.py`, `discord_notifier.py`,
`telegram_sender.py`, `report_generator.py` form the **live** reporting
pipeline (Discord/Telegram messages, not related to the backtest reporter)
— not read in depth this session; out of scope for backtest-platform work
unless a later phase needs to touch live notifications.

## 12. Existing tests

None. No `tests/` directory, no `pytest` config beyond the dependency being
listed, no test files found anywhere in `src/`.

## 13. Runtime / deployment

- pm2-managed: `Stock_ML` (the main daemon, `run.sh --daemon`),
  `CoinOne_Order`, `CoinOne_Proxy`, `coinone-rss-bot`, `pm2-dashboard`
  (unrelated Node process monitor), `PlungeDetector` (currently stopped).
- Streamlit apps (`app.py`, `dashboard.py`) are launched manually via
  shell scripts, not pm2-managed — confirmed no `streamlit run` reference
  anywhere in pm2 configs.
- No CI, no Dockerfile found, no deployment automation beyond pm2 +
  shell scripts.

## Existing working features (preserve as-is)

- Signal scoring formulas (momentum/volatility/drawdown/fundamentals/
  sentiment weighted sum) — per master-prompt rule #20, don't change
  behavior without documenting/testing the difference.
- Monthly/weekly rebalancing, Top-N equal-weight selection.
- Full backtest pipeline (engine/portfolio/performance/reporter), CLI,
  HTML/CSV export, Streamlit GUI — all working end-to-end as of this
  session.
- Macro LSTM regime detector (live path).
- KR order-execution code (`KROrderExecutor`/`KiwoomBroker`) — functionally
  complete, just not scheduled.
- DuckDB storage layer and the CSV->DuckDB load mechanism (aside from
  Finding 1/2 above).

## Existing features requiring correction

1. Daemon never checkpoints DuckDB (Finding 1) — **fixed this session** in
   the two snapshot-copy call sites; the daemon's own connection-handling
   still never checkpoints, which is fine for correctness now (readers
   copy the WAL too) but means `stockml.duckdb.wal` will grow unbounded for
   the life of the daemon process — worth a periodic `CHECKPOINT` call in
   the daemon itself eventually (not urgent, not done this session).
2. `RUN_FETCH=False` freezes live price data (Finding 2) — not fixed,
   needs a user decision (re-enable fully, or scope to specific tickers).
3. No minimum holding period / rebalance hysteresis (Section 5) — directly
   causes the user's reported short-holding-period behavior.
4. `KiwoomBroker.get_current_price` doesn't respect `paper_trading` the way
   the other three methods do (Finding 4).
5. Sharpe/Sortino omit a risk-free rate (Section 8) — not wrong, just an
   assumption that should be stated, not silently changed.
6. Returns are price-return only, dividends fetched but unused (Section 7).

## Missing functionality (per master prompt scope)

- Portfolio Construction Engine: only equal-weight exists. No
  signal-weighted, inverse-vol, risk-parity, or mean-variance allocation.
- Risk Engine: no correlation/covariance matrix, no risk contribution, no
  stress testing, no exposure monitoring beyond what `PerformanceReport`
  already computes (vol, drawdown, VaR/CVaR — already present, see Section 8).
- No walk-forward validation, no strategy comparison across allocation
  methods, no experiment persistence/run history.
- KR live order execution not scheduled (Finding 3); no GUI trade
  confirmation step; no GUI strategy-preset configuration beyond the
  backtest's own sidebar params.
- No `tests/` directory at all.

## Proposed additions

See `implementation_plan.md` for the phased breakdown.

## Potential breaking changes to watch for

- Any change to `_score_stocks`'s formula affects both live
  `decision_maker.py` and `backtest/engine.py` (they're independently
  duplicated, not shared) — a fix applied to one does not automatically
  apply to the other. This duplication itself is worth resolving in Phase
  1 (shared signal-scoring module) rather than patching both copies
  forever.
- Adding a minimum-holding-period mechanism changes historical backtest
  results for any existing saved reports — expected and desired, but worth
  flagging since "don't silently change existing behavior" is master-prompt
  rule #20.
- Re-enabling `RUN_FETCH` (if ever decided) changes the live system's
  outbound network behavior (58 tickers x yfinance calls every 30 min) —
  a infra/cost decision, not purely a code change.
