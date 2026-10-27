# Implementation Plan — Quantitative Portfolio Research & Risk Management Platform

Status: Stage 0 (Audit) complete. Phase 0.5 data backfill done (2026-10-28); validation harness next. Phase 1 not started.

Source: master prompt provided by the user, scoped against
`architecture_audit.md` and `financial_assumptions.md` findings.

## Execution approach (deviating from the master prompt's literal agent setup)

The master prompt specifies 3 fixed subagents (Quant Finance / Data &
Backtesting / Streamlit) coordinated by an orchestrator. In practice here:
most of Phase 1-3 is **tightly coupled** (portfolio construction, risk
engine, and backtesting execution all share the same data model and must
agree on interfaces before any of them can be built in parallel) — the
master prompt's own coordination rules say to prefer sequential
implementation for tightly-coupled financial modules and parallelize only
once interfaces are stable. So:

- Phases 1-3: built sequentially, by me directly, in this session/future
  sessions — not split across simulated subagent personas. Splitting
  tightly-coupled financial math across fresh-context agents who can't see
  each other's in-progress work risks exactly the kind of interface drift
  the master prompt warns against.
- Phase 4 (Streamlit, 6 pages): genuinely parallelizable once Phase 1-3
  data contracts are frozen — this is where spawning agents for
  independent pages makes sense, if the user wants that at the time.
- Every phase ends with the acceptance criteria the master prompt itself
  defines, run for real (not claimed) before moving on.

## Phase 0 — Audit (this stage)

- [x] Repository structure, dependencies, signal/regime/backtest logic
      mapped (`architecture_audit.md`).
- [x] `./run_backtest.sh` argument support verified directly against argparse.
- [x] Benchmark audit (default, configurability, alignment, dividends,
      silent-fallback check) — done, one real bug found and fixed (SPY/QQQ
      missing from DB), one real bug found and fixed (DB staleness via
      uncheckpointed WAL), two assumptions documented, not silently changed
      (dividends, risk-free rate).
- [x] Financial metric formula audit, suspicious-figures investigation
      (`financial_assumptions.md`).
- [x] `docs/architecture_audit.md`, `docs/implementation_plan.md`,
      `docs/financial_assumptions.md` created.

**Open decisions needing user confirmation before Phase 1 starts** (asked
separately, not assumed):

1. Dividend handling — switch to adjusted-close total return, or leave as
   price-return with the assumption documented?
2. Risk-free rate — add to Sharpe/Sortino (needs a rate source + lookback
   convention decision), or leave as return/vol ratio?
3. Minimum holding period — this is the direct fix for the "3-day sell"
   complaint. Needs a concrete rule, e.g.: "a position can't be sold at a
   rebalance unless held >= N rebalance cycles" or "existing holdings get
   a score bonus so marginal newcomers don't bump them." Which mechanism,
   and what N?
4. `RUN_FETCH=False` — leave disabled (manual/ad hoc data updates only,
   like the SPY/QQQ fetch done this session) or re-enable for all 58
   tickers on the existing 30-min schedule?
5. Kiwoom paper-trading wiring — the master prompt's Phase 3 doesn't
   explicitly include this, but the user's own plain-language ask prioritized
   it ("generate actual trading logs... FIRST"). Confirm: wire
   `KROrderExecutor` into a scheduled job (or a GUI-triggered action) so paper
   trades actually get generated and logged, with a GUI confirmation step
   before each batch executes? This stays paper-only per master-prompt rule
   #19 (no live trading) regardless.

## Phase 0.5 — Data backfill + validation harness (inserted 2026-10-28, before Phase 1)

Why: the DB held ~1yr of prices (2024-11 -> 2025-11). A 250-day Sharpe has
standard error ~1.0, so nothing built on it (allocators, ML) could be judged.

Done:
- [x] `src/research/backfill_prices.py` — 58 config tickers, 2010-01-04 ->
      2026-10-27, 229,921 rows, adjusted OHLCV (yfinance auto_adjust=True, so
      closes are split+dividend adjusted; this resolves open decision 1 for
      prices: total-return basis). Writes `data/research/research.duckdb`
      (snapshot of live DB with `price_daily_full` replaced). Live DB is never
      written (daemon holds the lock).
- [x] `run_backtest.py --db <path>` flag.
- [x] First long run (2012-01-01 -> 2025-10-31, monthly, top-10, SPY):
      CAGR 29.9% vs SPY 14.9%, Sharpe 1.10, MDD -36%, 974d max DD duration.

**Read this result with suspicion, not celebration.** The universe is 58
hand-picked, today-known winners (NVDA, AVGO, ...): survivorship/look-ahead
in universe selection almost certainly explains most of the excess return.
Over 2012-2025 only price momentum is active (news exists for 30 days,
fundamentals ~1yr). So this measures "momentum on hindsight-picked mega-caps",
not a deployable edge. Treat as an upper bound.

Next (in order):
- [ ] Walk-forward harness with embargo (expanding train, out-of-sample test).
- [x] Factor IC baseline `src/research/factor_ic.py` (2011-2026, 189 monthly
      dates, 56 stocks): mom_12_1 mean IC 0.042, t=2.35, 57% hit rate, yearly
      IC -0.04..+0.19; mom_12 0.041; 1m reversal 0.005 (no signal). This is
      the bar any ML model must beat out of sample.
- [x] Universe bias check `src/research/universe_bias.py`: mom_12_1 IC is
      robust to universe choice (300 random 30-stock subsets: mean 0.041,
      p5-p95 0.021-0.059, 0% <= 0; ex top-10 winners: 0.040). The hindsight
      universe mostly inflates backtest *returns* (level), not the ranking
      signal. Remaining bias: all names are current survivors; no delisted
      stocks. Quintile/factor-turnover stats still to add.
- [x] Ridge walk-forward `src/research/ridge_wf.py` (6 close-only features,
      expanding window, 1-sample embargo, 151 OOS dates 2014-2026): ridge
      OOS IC 0.029-0.030 vs plain 12-1 momentum 0.040 on the same dates;
      paired diff -0.010 (t -0.8 to -0.9), i.e. NOT better. Not yet tested:
      volume/beta features, LightGBM, longer universe. Decision: ML adds
      nothing on this data so far; do not build further ML until the
      universe/feature set expands.
- [x] Pool-trained walk-forward `src/research/pool_wf.py` (train on 448 S&P 500
      names, ~110k rows/window, weekly samples, 25d purge; test only on the 56;
      129 OOS dates 2013-2026): baseline mom_12_1 IC 0.048; ridge 5y-roll
      -0.021, HGB 5y-roll -0.024, ridge expanding 0.001, HGB expanding -0.019;
      paired diff vs baseline -0.05 to -0.07 (t -1.7 to -2.4), i.e. WORSE,
      not just flat. Dropping the pool's top-10% winners changes nothing
      (-0.022). Pool-trained signal does not transfer to the 56 mega-caps.
      Conclusion: ML stays out of the portfolio; revisit only with a larger
      cleaner dataset (point-in-time, delisted names, volume/fundamental features).
- [ ] Fix scoring bugs found in review: vol term negligible (scale mismatch),
      no 12-1 skip-month, revenue-CV rewarded (wrong sign), no cross-sectional
      z-score/winsorize, 60-day filing lag for fundamentals.
- [ ] Bias controls: universe sensitivity (random 30-ticker subsets, ex-top-
      winners), cost sensitivity (0/5/10/25 bps), deflated Sharpe.
- [ ] Regime: replace macro vote-count (Risk-On was -16% in the 1yr run) with
      market-based signals (SPY vs 200d, VIX, curve) or 2-3 state HMM; only
      keep if forward returns are monotone across regimes.
- [ ] Tests dir started here.

## ML policy
Rank-based cross-sectional model (ridge/elastic-net baseline first, tree
model second) predicting 21d forward excess return, purged walk-forward.
Adopted only if rank-IC beats the plain 12-1 momentum baseline out of
sample, after costs, stable across years. LSTM next-day return prediction is
dropped (dir. accuracy 50.3%, worse than naive RMSE, see reports/lstm_eval.md).
ML never gets sole control: it feeds the score; portfolio constraints, vol
target and holding-period buffer stay rule-based.

## Phase 1 — Portfolio Management Engine (Subagent A role, done directly)

Target files (new, under `src/portfolio/`):
- `models.py` — Portfolio/Position dataclasses (Section 1.1 of master prompt).
- `allocation.py` — equal-weight (already exists in `engine.py`, extract
  and reuse rather than reimplement), signal-weighted, inverse-volatility,
  risk-parity (via `scipy.optimize`), constrained mean-variance (`scipy`,
  not `cvxpy` unless a constraint genuinely needs it — avoid the extra
  dependency per the master prompt's own "avoid unnecessary dependencies"
  rule until proven necessary).
- `constraints.py` — max/min position weight, max sector exposure, max
  holdings, min cash, max turnover, long-only. Infeasible constraints raise,
  never silently violate (master prompt requirement).
- `accounting.py` — reuse `PortfolioSimulator` from `backtest/portfolio.py`
  as the base (it already does cash/position/commission/slippage
  accounting correctly per this session's testing) rather than building a
  parallel accounting system — extend it with the new allocation methods'
  output format instead of replacing it.

Acceptance criteria: as specified in the master prompt Section 4 (weights+
cash reconcile to 100%, NAV reconciles, constraints respected, no negative
cash long-only, deterministic unit tests per allocation method, equal-weight
reproduces the existing baseline).

New dependency: `scipy` (not yet installed — add via poetry).

## Phase 2 — Risk Intelligence (Subagent A role)

Target files (new, under `src/risk/`):
- `statistical.py` — rolling/annualized vol, covariance/correlation
  matrix, portfolio vol, beta, tracking error (some of this already exists
  per-asset in `performance.py` — extract shared pieces rather than
  duplicate).
- `downside.py` — drawdown series/duration, VaR/CVaR 95/99 (already
  implemented correctly in `performance.py` for the single-portfolio case;
  generalize to the new multi-asset portfolio engine's output).
- `contribution.py` — marginal/component risk contribution
  (`RC_i = w_i*(Sigma@w)_i / sigma_p`), with a reconciliation check
  (`sum(RC_i) ≈ sigma_p`) as a runnable assertion, not just a comment.
- `stress.py` — historical replay (only for periods with sufficient real
  data — the DB's actual coverage is ~1yr per `architecture_audit.md`
  Finding 2, so historical replay will be limited until `RUN_FETCH` /data
  backfill decisions land) and hypothetical shock scenarios, clearly
  labeled as separate (master prompt requirement).

Acceptance criteria: formulas checked against hand-computed reference
cases (small N, e.g. 3-asset portfolio with known covariance), risk
contributions reconcile within numerical tolerance.

## Phase 3 — Backtesting & Strategy Research (Subagent B role)

- Preserve existing signals/regime/rebalancing as-is (master prompt
  3.1) — the scoring formula does not change in this phase; only the
  **holding-period/hysteresis mechanism** changes (per the open decision
  above), and that change is applied identically to the formula everywhere
  it's duplicated (`decision_maker.py` and `engine.py` both, closing the
  "fix applied to one copy only" risk noted in the audit).
- Signal-to-portfolio interface: formalize the dict shape already
  implicitly used (`stock_id`, `score`) into the schema the master prompt
  specifies, consumed by the new Phase 1 allocation engine instead of
  `engine.py::_build_target_weights`'s hardcoded equal-weight logic.
- Transaction cost model: already implemented (slippage+commission) —
  extend to report gross vs. net separately (currently only net is shown,
  per `financial_assumptions.md`).
- Benchmark engine: SPY default + QQQ secondary now both work (fixed this
  session); extend selector to also support "Equal-weight eligible
  universe" as a computed (non-ticker) benchmark option, explicitly
  requested by the user in an earlier message this session.
- Strategy comparison: run the existing formula + each Phase 1 allocation
  method + buy-and-hold benchmark over identical windows, tabulate side by
  side. This is largely a new orchestration layer over the already-working
  `BacktestEngine`, not a rewrite of it.
- Walk-forward validation: rolling/expanding windows, parameter freezing
  before the out-of-sample segment. New code; nothing like this exists
  today.
- Performance attribution: stock-level and regime-level attribution
  already exist (`PerformanceCalculator._regime_attribution`,
  `get_round_trips()` for stock-level). Sector attribution needs sector
  classifications, which don't currently exist anywhere in the data model
  — needs a sector-mapping table before this can be implemented honestly
  (master prompt rule: don't claim attribution without the underlying data).

## Phase 4 — Streamlit Dashboard (Subagent C role)

The existing `backtest/dashboard.py` already covers a meaningful slice of
"Page 1 — Overview" and "Page 6 — Trade Ledger" from the master prompt's
spec. Rather than rebuild from scratch (master prompt's own instruction:
don't rebuild unless demonstrably unsuitable), restructure it into a
Streamlit multipage app (`src/backtest/pages/` + a slim entry file) once
Phase 1-3 data contracts exist to build the remaining 4 pages against:
Signal Explorer, Portfolio Builder, Risk Analytics, Backtest Lab (the
"explicit Run button, no silent execution on control change" requirement
is already satisfied by the current design — `st.session_state` gating,
confirmed working this session).

This phase is the best candidate for parallel subagent work once Phase 1-3
interfaces are frozen, since each page is largely independent given a
stable data layer.

## Phase 5+ — Data engineering, testing, docs (ongoing, not a separate gate)

- `tests/` directory: none exists. Start one in Phase 1 (not deferred to
  the end) — every new allocation/risk function gets a deterministic unit
  test as it's written, per the master prompt's own "ponytail"-compatible
  rule that non-trivial financial logic needs a runnable check.
- Reproducibility (run ID, config snapshot, data fingerprint): add once
  Phase 3's backtest orchestration layer exists — premature before that.
- Remaining `docs/` files (system_architecture.md, portfolio_methods.md,
  backtesting_methodology.md, risk_metrics.md, data_dictionary.md,
  testing_report.md, limitations.md): written incrementally as each phase
  lands, not upfront — writing methodology docs for code that doesn't
  exist yet produces docs that lie.

## Checklist (updated as work completes)

- [x] Stage 0 audit
- [ ] Open decisions 1-5 confirmed with user
- [x] Phase 0.5: price backfill (harness + factor research still open)
- [ ] Phase 1: portfolio data model + allocation methods + constraints + accounting
- [ ] Phase 2: risk intelligence
- [ ] Phase 3: backtesting/strategy research upgrades
- [ ] Phase 4: dashboard multipage expansion
- [ ] Ongoing: tests written alongside each phase
