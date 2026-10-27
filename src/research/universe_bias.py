"""
How much of the momentum IC depends on which stocks are in the universe?
Compares full-universe mean IC of mom_12_1 against (a) 300 random 30-stock
subsets and (b) the universe with the 10 best long-run performers removed.

Usage: poetry run python src/research/universe_bias.py
"""
import numpy as np
import pandas as pd

from factor_ic import H, factors, load_prices, ROOT


def mean_ic(f: pd.DataFrame, fwd: pd.DataFrame, cols, dates, min_n=20) -> float:
    ics = []
    for t in dates:
        a, b = f.loc[t, cols], fwd.loc[t, cols]
        ok = a.notna() & b.notna()
        if ok.sum() >= min_n:
            ics.append(a[ok].rank().corr(b[ok].rank()))
    return float(np.mean(ics))


if __name__ == "__main__":
    px = load_prices(str(ROOT / "data/research/research.duckdb"))
    f, fwd = factors(px)["mom_12_1"], px.shift(-H) / px - 1
    dates = f.index[252::H]
    cols = list(px.columns)
    full = mean_ic(f, fwd, cols, dates)

    rng = np.random.default_rng(0)
    subs = np.array([mean_ic(f, fwd, list(rng.choice(cols, 30, replace=False)), dates) for _ in range(300)])

    tot = (px.iloc[-1] / px.bfill().iloc[0]).sort_values(ascending=False)
    ex = mean_ic(f, fwd, [c for c in cols if c not in tot.index[:10]], dates)

    print(f"full universe ({len(cols)}):          mean IC {full:.4f}")
    print(f"random 30-stock subsets (n=300):  mean {subs.mean():.4f}  p5 {np.percentile(subs,5):.4f}  "
          f"p95 {np.percentile(subs,95):.4f}  share<=0: {(subs<=0).mean():.0%}")
    print(f"ex top-10 long-run winners ({len(cols)-10}): mean IC {ex:.4f}  (dropped: {', '.join(tot.index[:10])})")
