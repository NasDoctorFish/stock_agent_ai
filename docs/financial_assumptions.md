# Financial Assumptions — Stock_ML

Documents every assumption baked into the current calculations, per the
master prompt's requirement that suspicious-looking figures be explained
rather than silently "fixed" or left ambiguous. See `architecture_audit.md`
Section 8 for the full formula table this expands on.

## Return basis

- **Price return, not total return.** `fetch_prices()` captures `Dividends`
  and `Stock Splits` from yfinance into `data/raw/price_daily.csv`, but
  every NAV/return calculation in `engine.py` and `performance.py` reads
  only the `close` column. Dividends are fetched and stored but never
  applied. For dividend-paying benchmarks (SPY ~1.3%/yr) this understates
  both strategy and benchmark total return by roughly the dividend yield,
  compounded over the window. Equity-only, zero-income instruments (most of
  the 56-ticker growth-tilted universe) are less affected. **Not corrected
  this audit** — a real methodology decision (switch to adjusted close, or
  explicitly add dividend cash flows to NAV) that changes every historical
  number once made.

## Risk-free rate

- **Sharpe and Sortino ratios do not subtract a risk-free rate.**
  `sharpe_ratio = CAGR / annualized_volatility`. This is a return/risk
  ratio, not the textbook `(return - rf) / vol`. In a near-zero-rate
  environment the difference is small; at current short-term rates
  (~4-5%) it matters. Document, don't silently "fix" — changing this
  changes every historical Sharpe/Sortino number and needs a rate source
  (the DB already ingests `FEDFUNDS`/`DGS10` via `macro_index_full`, so a
  source exists if this gets addressed in Phase 2).

## Annualization

- 252 trading days/year (`TRADING_DAYS = 252` in `performance.py`) —
  standard US-market convention. Applied consistently across volatility,
  Sharpe, Sortino, information ratio, alpha annualization.
- CAGR uses calendar days (`years = days/365.25`), not trading days — also
  standard, and correctly distinguished from the 252-day convention used
  elsewhere (not a mismatch, just two different correct conventions for
  two different purposes: CAGR is a calendar-time compounding rate,
  volatility/Sharpe are trading-day-frequency statistics).

## Regression Alpha vs. excess return

- `excess_return = total_return - benchmark_total_return` — simple
  arithmetic difference, labeled "Excess Return" in every output
  (console/HTML/GUI).
- `alpha` — true OLS regression intercept (`alpha_daily = mean(y) -
  beta*mean(x)`, annualized `*252`), labeled "Alpha" separately. These are
  **not conflated** anywhere in the current code — master-prompt rule #10
  is already satisfied. They will differ substantially when beta deviates
  from 1.0 or when volatility is high relative to the window length (seen
  directly this session: a 5-month window with beta=1.54 produced
  Alpha=31-75% vs. excess return in the 6-35% range across different runs —
  both numbers are individually correct, they just answer different
  questions, and the gap is a real small-sample/high-beta artifact, not a
  bug).

## Inverse-volatility vs. risk parity

- Not yet implemented (Phase 1). When implemented: inverse-volatility
  weighting (`w_i = (1/sigma_i) / sum(1/sigma_j)`) must be labeled exactly
  that, never "risk parity" — true risk parity requires the full covariance
  matrix (equal *risk contribution*, not equal inverse-vol, which only
  equals risk parity when all pairwise correlations are identical). Master
  prompt rule #9 explicitly calls this out; noting it here so the
  distinction is enforced from the first implementation, not retrofitted.

## Small-sample fragility (the suspicious-metrics finding)

Investigated directly: Profit Factor ≈49, Win Rate 75%/29 trades, and
large annualized-Alpha figures referenced in the master prompt were
reproduced this session using short backtest windows (e.g. 5-month,
`--start 2025-06-01`). Root cause confirmed as **sample size and window
length**, not a calculation error:

- Profit Factor = gross_profit / gross_loss over a small number of round
  trips (~20-30). A handful of small losses against several large wins
  trivially produces ratios in the 10s-50s range. This is mathematically
  correct and simultaneously nearly meaningless as a robustness signal at
  that sample size.
- CAGR/Alpha annualize a short absolute return by `^(1/years)` /
  `*252` respectively — a strong 5-month run gets amplified hard when
  projected to an annual rate. Also correct, also fragile.

**Recommendation** (for Phase 3.7, walk-forward validation): surface a
minimum-sample warning (e.g. "fewer than ~30 round trips — treat ratios as
indicative, not robust") rather than hiding or rounding these figures.
Never suppress a real computed number because it looks surprising — the
fix is context, not censorship, per master-prompt rule #3/#4.

## Regime detection: live vs. backtest divergence

Two separate, independently-tuned regime classifiers exist (full detail in
`architecture_audit.md` Section 4): a trained LSTM for live US regime, and
a 6-signal heuristic for backtest US regime. **A backtest run today does
not reproduce what the live LSTM-driven system would have decided on the
same historical dates.** This is by the engine's own explicit design
("No live API calls" — the LSTM needs iterative training and isn't
reproducible/replayable the same way), not an oversight, but it means
backtest results should be described as "what this rule-based heuristic
regime + fixed scoring formula would have done," not "what Stock_ML would
have done" — those are different claims. Any dashboard surfacing backtest
results should carry this caveat.

## Transaction costs

- Slippage: 0.001 (10bps) one-way, applied multiplicatively to execution
  price (`price * (1 ± slippage_pct)`) — correct direction (worse price
  for both buys and sells).
- Commission: 0.00015 (1.5bps) of gross trade value, charged on both buys
  and sells.
- Both are configurable per `BacktestEngine` constructor args and now
  exposed in the GUI's "Advanced" expander (added this session).
- **Gross vs. net performance are not currently reported separately** —
  the master prompt's Phase 3.4 asks for both; today only net (after
  cost) performance is shown. Flagged as a Phase 3 gap, not a bug.

## Look-ahead / leakage

Confirmed no look-ahead in the backtest's price selection
(`_prices_on_date` strictly filters `date < rebal_date`) or in the
regime/scoring SQL queries (`engine.py`'s own stated design principle,
verified by reading every query this session: all use `< rebal_date`
bounds). Walk-forward validation (Phase 3.7) is still needed for anything
beyond this single-pass backtest — there is currently no mechanism to
prevent a human from, say, tuning `top_n` or the scoring weights by eye
against the full historical window and calling that "validated."
