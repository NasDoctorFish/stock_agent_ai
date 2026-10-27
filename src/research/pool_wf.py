"""
Pool-trained walk-forward: train on the S&P 500 pool (minus the 56 config
tickers), test ONLY on the 56. Compare against plain 12-1 momentum.

- Weekly samples (STEP=5 trading days); target = 21d forward return.
- Features and target are rank-normalised per date over the FULL universe
  (pool + 56), IC is then measured on the 56 only.
- Purge: training samples end >= PURGE samples before the test date, so no
  training target window reaches the test date. Test dates are every PURGE
  samples, so test windows do not overlap.
- Windows: rolling 5y (252 samples) and expanding.
- Models: ridge, HistGradientBoosting (same family as LightGBM; sklearn, no new dep).
- Sensitivity: HGB rolling with the pool's top-10% long-run winners removed.

Usage: poetry run python src/research/pool_wf.py
"""
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from factor_ic import H, ROOT, load_prices
from ridge_wf import features

STEP, PURGE, ROLL, FIRST = 5, 5, 252, 150
DB = str(ROOT / "data/research/research.duckdb")


def panel() -> tuple:
    test_px = load_prices(DB)
    pool = duckdb.connect(DB, read_only=True).execute("SELECT CAST(date AS DATE) d, stock_id, close FROM price_pool").fetchdf()
    pool_px = pool.pivot(index="d", columns="stock_id", values="close").sort_index()
    px = pd.concat([test_px, pool_px], axis=1).sort_index()
    px = px.loc[:, ~px.columns.duplicated()]
    feats, fwd = features(px), px.shift(-H) / px - 1
    rows = []
    for t in px.index[252:-H:STEP]:
        d = pd.DataFrame({k: v.loc[t] for k, v in feats.items()})
        d["y"] = fwd.loc[t]
        d = d.dropna().rank(pct=True) - 0.5
        d["date"], d["is_test"] = t, d.index.isin(test_px.columns)
        rows.append(d.rename_axis("stock").reset_index())
    return pd.concat(rows, ignore_index=True), px


def ic(a, b):
    return spearmanr(a, b)[0]


if __name__ == "__main__":
    P, px = panel()
    X = [c for c in P.columns if c not in ("stock", "y", "date", "is_test")]
    dates = sorted(P.date.unique())
    tot = px.iloc[-1] / px.bfill().iloc[0]
    pool_names = P[~P.is_test].stock.unique()
    winners = set(tot[pool_names].sort_values(ascending=False).index[: len(pool_names) // 10])
    print(f"{len(dates)} sample dates, {P.stock.nunique()} stocks, pool rows {(~P.is_test).sum()}")

    def hgb():
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                             min_samples_leaf=500, l2_regularization=10.0, random_state=0)

    res, ntr = {}, []
    for i in range(FIRST, len(dates), PURGE):
        te = P[(P.date == dates[i]) & P.is_test]
        if len(te) < 30:
            continue
        hi = dates[i - PURGE]
        for win, lo_i in (("roll", max(0, i - PURGE - ROLL)), ("exp", 0)):
            tr = P[(P.date >= dates[lo_i]) & (P.date <= hi) & ~P.is_test]
            ntr.append(len(tr))
            for name, m, trn in (("ridge", Ridge(alpha=100), tr), ("hgb", hgb(), tr),
                                 ("hgb-exWinners", hgb(), tr[~tr.stock.isin(winners)])):
                if name == "hgb-exWinners" and win == "exp":
                    continue
                m.fit(trn[X], trn["y"])
                res.setdefault(f"{name} {win}", {})[dates[i]] = ic(m.predict(te[X]), te["y"])
        res.setdefault("mom_12_1 (baseline)", {})[dates[i]] = ic(te["mom_12_1"], te["y"])
    R = pd.DataFrame(res)
    print(f"{len(R)} OOS dates {R.index[0].date()} -> {R.index[-1].date()}; median train rows {int(np.median(ntr))}\n")
    base = R["mom_12_1 (baseline)"]
    for c in R:
        d = R[c] - base
        extra = "" if c.startswith("mom") else f"  vs base {d.mean():+.4f} (paired t {d.mean()/(d.std()/np.sqrt(len(d))):.2f})"
        print(f"{c:22s} IC {R[c].mean():.4f}  t {R[c].mean()/(R[c].std()/np.sqrt(len(R))):.2f}  hit {(R[c]>0).mean():.0%}{extra}")
    print("\nOOS mean IC by year:\n", R.groupby(R.index.year).mean().round(3).to_string())
