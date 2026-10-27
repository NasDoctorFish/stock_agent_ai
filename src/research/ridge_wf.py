"""
Walk-forward ridge for cross-sectional ranking vs the plain 12-1 momentum baseline.

Row = (date t, stock). Features: close-only, rank-normalised per date to
[-0.5, 0.5]. Target: rank-normalised 21d forward return. Dates are sampled
every H=21 trading days; at test date i the model is fit on dates <= i-2
(one extra sample of embargo beyond the non-overlap), expanding window,
refit every date. Metric: Spearman IC per test date, same dates for all models.

Usage: poetry run python src/research/ridge_wf.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from factor_ic import H, ROOT, load_prices

MIN_TRAIN = 36   # sample dates before the first prediction
ALPHAS = [10, 100, 1000]  # reported all; no picking on test


def features(px: pd.DataFrame) -> dict:
    r = px.pct_change()
    return {
        "mom_12_1": px.shift(21) / px.shift(252) - 1,
        "mom_6_1":  px.shift(21) / px.shift(126) - 1,
        "mom_3m":   px / px.shift(63) - 1,
        "rev_1m":   -(px / px.shift(21) - 1),
        "vol_60":   -r.rolling(60).std(),
        "dd_252":   px / px.rolling(252).max() - 1,
    }


def rank_norm(row: pd.Series) -> pd.Series:
    return row.rank(pct=True) - 0.5


def build_panel(px: pd.DataFrame) -> pd.DataFrame:
    feats, fwd = features(px), px.shift(-H) / px - 1
    rows = []
    for t in px.index[252::H]:
        d = pd.DataFrame({k: rank_norm(v.loc[t].dropna()) for k, v in feats.items()})
        d["y"] = rank_norm(fwd.loc[t].dropna())
        d = d.dropna()
        d["date"] = t
        rows.append(d.reset_index(names="stock"))
    return pd.concat(rows, ignore_index=True)


def ic(pred: np.ndarray, y: np.ndarray) -> float:
    return spearmanr(pred, y)[0]


def summarize(s: pd.Series) -> str:
    n = len(s)
    return (f"IC {s.mean():.4f}  std {s.std():.3f}  t {s.mean()/(s.std()/np.sqrt(n)):.2f}  "
            f"hit {(s>0).mean():.0%}  n={n}")


if __name__ == "__main__":
    px = load_prices(str(ROOT / "data/research/research.duckdb"))
    P = build_panel(px)
    X = [c for c in P.columns if c not in ("stock", "y", "date")]
    dates = sorted(P.date.unique())
    # y at the last sampled dates has NaN fwd -> already dropped by dropna
    res = {"mom_12_1 (baseline)": {}, **{f"ridge a={a}": {} for a in ALPHAS}}
    coefs = {a: [] for a in ALPHAS}
    for i in range(MIN_TRAIN + 2, len(dates)):
        tr = P[P.date <= dates[i - 2]]
        te = P[P.date == dates[i]]
        res["mom_12_1 (baseline)"][dates[i]] = ic(te["mom_12_1"].values, te["y"].values)
        for a in ALPHAS:
            m = Ridge(alpha=a).fit(tr[X], tr["y"])
            coefs[a].append(m.coef_)
            res[f"ridge a={a}"][dates[i]] = ic(m.predict(te[X]), te["y"].values)
    R = pd.DataFrame(res)
    print(f"{len(R)} OOS dates {pd.Timestamp(R.index[0]).date()} -> {pd.Timestamp(R.index[-1]).date()}, features {X}\n")
    for c in R:
        print(f"{c:22s}", summarize(R[c]))
    base = R["mom_12_1 (baseline)"]
    print()
    for a in ALPHAS:
        d = R[f"ridge a={a}"] - base
        print(f"ridge a={a} minus baseline: mean {d.mean():+.4f}  paired t {d.mean()/(d.std()/np.sqrt(len(d))):.2f}")
    print("\nmean coefs (a=100):", dict(zip(X, np.mean(coefs[100], axis=0).round(3))))
    print("\nOOS mean IC by year:\n", R.groupby(pd.to_datetime(R.index).year).mean().round(3).to_string())
