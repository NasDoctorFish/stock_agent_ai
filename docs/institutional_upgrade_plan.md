# Institutional research upgrade — first pass (Phases 0-4 + E1-E9)

## Context
Question to answer: does ML add predictive value beyond momentum/factors, survive OOS, and survive costs? Existing research (`src/research/`: factor_ic, ridge_wf, pool_wf, universe_bias, ablation, familiarity, val_followup) is ad hoc scripts with different date grids, so results are not comparable. Build a shared harness + factor engine under `src/research/` (extend, don't rebuild), keep old scripts and `reports/*` untouched.

Scope chosen by user: Phases 0-4 and experiments E1-E9. Deferred to a later pass after review: Phase 5 optimizer (only a simple turnover-buffer portfolio now, reusing `val_followup.buffered_spread`), Phase 7 Streamlit, E10. Deps approved: `lightgbm`, `statsmodels` (poetry add). Sectors: yfinance current sector snapshot, cached to versioned CSV, flagged non-PIT.

## Facts from audit (constrain design)
- Prices: `price_daily_full` 56 stocks (+SPY,QQQ benchmarks) 2010→2026, adjusted; `price_pool` ~448 S&P names (survivors). No delisted names, no historical membership → every experiment labelled "survivor-biased".
- Fundamentals: income-statement only, fiscal_date 2024-02→2026-09, no filing date, no balance sheet. **P/B, ROE, ROIC, EV/EBITDA impossible** → factor registry marks them `unavailable` (not NaN-filled). Possible: earnings yield, P/E, operating margin, YoY earnings growth (needs ≥5 quarters → starts ~2025-05+).
- Existing tests: only `src/stock_ml/tests/test_platform.py`; no repo `tests/`. Dashboard: `src/backtest/dashboard.py` (Streamlit) — untouched this pass.
- Existing reusable: `ridge_wf.rank_norm/ic`, `factor_ic.load_prices/H/ROOT`, `familiarity.price_familiarity/valuation/ttm_eps`, `ablation.boot_ci/spread/walk_forward` (refactor into harness, keep old callers working), `src/backtest/performance.py` for CAGR/Sharpe/MDD.
- Known prior numbers are on different grids (127 vs 151 vs 129 dates, weekly vs monthly); report will label each.

## Deliverables / file layout (all new unless noted)
1. **Phase 0** `docs/institutional_upgrade_plan.md` — architecture, files, validation strategy, acceptance criteria (this plan, expanded). First action after approval.
2. **Phase 1 — `src/research/pitdata/` (renamed: avoids clash with top-level data/)**
   - `pit.py`: `load_universe()` (config tickers minus benchmarks, documented as survivor set), `availability(fiscal_date, lag)` single function used everywhere, `load_fundamentals(lag)`.
   - `quality.py`: missing-data diagnostics, price sanity (non-positive, >50% daily jumps vs split), duplicate keys.
   - `manifest.py`: dataset version = sha256 of source tables + row counts + date ranges + git commit, written to `reports/research/manifest_<hash>.json`.
   - EPS lag 45/60/90 is a parameter threaded through factor build (E9).
   - Tests: EPS unavailable before fiscal_date+lag; no feature uses row t+1; manifest deterministic.
3. **Phase 2 — `src/research/factors/`**
   - `registry.py`: factor name → fn, `status` (available/unavailable+reason), family (momentum/value/quality/risk).
   - Momentum: mom_12_1, mom_6_1, rev_1m, risk-adj mom (mom_12_1 / 252d vol). Value: ey, pe (P/B, EV/EBITDA = unavailable). Quality: op_margin_ttm, earnings_growth_yoy (ROE/ROIC unavailable). Risk: vol_60, beta_252 (vs SPY), dd_252, amihud/dollar-volume liquidity (volume exists).
   - `transform.py`: per-date winsorize (1/99 cross-sectional percentile, parameter-free so no train leakage), z-score/rank, sector-demean; `Preprocessor` class with `fit(train)`/`transform` for any learned params (e.g. fill medians, scaling) — fit on training dates only. Missing fundamentals stay NaN + `has_x` mask; models receive them only via explicit masks, never fabricated values (existing ablation 0-fill is replaced by mask/subset).
   - `sectors.py`: fetch via yfinance once → `data/research/sectors_v1.csv`; flagged non-PIT.
   - `evaluate.py`: per-factor Rank-IC, ICIR, yearly IC, sector IC, partial IC (vs mom_12_1+rev_1m), turnover-adjusted quintile spread.
4. **Phase 3 — `src/research/models/`**
   - Track A `factor_model.py`: fixed equal-weight composite of z-scored momentum + value + quality − risk, weights from config (no fitting); optional ridge-fitted weights as variant.
   - Track B `direct.py`: Ridge, sklearn HGB (existing), LightGBM; targets `raw` and `sector_rel` (fwd ret minus sector mean), same dates/universe.
   - Track C `residual.py`: expanding time-respecting cross-fit — for each outer test date, factor forecast on training rows comes from fits on strictly earlier folds only (K chronological folds with purge; fold k's forecast uses factor model fit on folds < k), residual = realized − out-of-fold factor forecast; ML fits residual; final = factor + λ·resid_pred; λ ∈ {0, .25, .5, 1} chosen by nested validation on the last block of the training window only; λ=0 always reported.
5. **Phase 4 — `src/research/harness.py`**
   - `ExperimentConfig` dataclass (seed, horizon H=21, window roll/expanding, universe, lag, target, embargo, cost bps) → JSON snapshot saved with results in `reports/research/<exp_id>/`.
   - Splitter: sample dates every 21d; train dates ≤ test−(purge) where purge ≥ label horizon in samples; nested chronological validation for hyperparams (alpha, λ) inside training only; **final holdout = last 24 months, evaluated once** after all selection, reported separately.
   - Common eval grid: every model in an experiment evaluated on identical dates/stocks (intersection), enforced by assertion.
   - Stats: mean IC, annualised ICIR, paired ΔIC vs baseline, Newey-West t (statsmodels, lags=2 for monthly overlap-free but serial dependence), stationary block bootstrap CI, yearly + sector stability, Holm correction across the experiment family (p-values per candidate vs baseline).
   - Portfolio metrics from IC predictions: quintile long-short and long-only top-quintile, gross/net at cost sensitivity {0,5,10,25} bps, turnover, buffered variant; CAGR/Sharpe/MDD.
6. **Phase 6 — `src/research/experiments/e1.py … e9.py`** + `run_all.py`; each writes config, table, limitations to `reports/research/`. E10 recorded as "not run: no historical membership data".
   - E1 mom-only; E2 multi-factor linear; E3 Ridge/HGB/LGBM direct; E4 mom+ML blend (z-score blend, weight by nested validation); E5 factor+residual ML; E6 sector-rel vs raw; E7 rolling vs expanding; E8 cost sensitivity; E9 lag 45/60/90 (valuation subset, all models on identical dates, labelled exploratory since ≤14 dates).
   - Price-only experiments use full 2016-2026 grid; anything touching fundamentals uses the short val-eligible grid, labelled separately in output.
7. **Phase 8 (partial, needed for acceptance)** `tests/research/` (pytest, already a dev dep): leakage (truncate-and-recompute equality for every factor), feature-target alignment, PIT availability, residual cross-fit (no fold uses its own or future data), split integrity (max train label end < test date − purge), determinism (same seed → identical output), cost accounting (turnover×bps hand-computed case). Report `docs/institutional_research_report.md` with successes and failures; reproduce commands per experiment.

## Execution order & gates
0 plan doc → 1 data (tests pass) → 2 factors (tests pass) → harness (Phase 4 core, since 3 depends on it) → 3 models → experiments → report. After each phase: run its tests, summarise done/tests/issues. Sequential, done directly (interfaces coupled; consistent with `docs/implementation_plan.md` decision). Subagent use only for an independent QA/leakage review of finished code at the end (Agent 5 role), not concurrent writers.

## Verification
- `poetry run pytest tests/research -q` green.
- `poetry run python -m src.research.experiments.run_all --quick` end-to-end; E1 mom_12_1 IC on the common grid cross-checked against `factor_ic.py` output (differences explained by date grid).
- Placebo: shuffled features → ΔIC≈0 inside harness test.
- Re-run twice → identical result hashes.
- Final report states plainly if ML fails to beat momentum and keeps the baseline.

## Risks to flag in the report
Survivor universe (56 hindsight mega-caps), ≤14 fundamental dates, sector snapshot non-PIT, holdout of 24 months is small (~24 monthly dates), multiple-testing across ~30 candidates, no delisting returns.
