"""
Cross-sectional factor IC baseline.

For each sampled date t (every H trading days, so forward windows do not
overlap): factor is computed from prices <= t, target is the return over
(t, t+H]. IC_t = Spearman(factor_t, fwd_ret_t) across stocks. Also reports
top-minus-bottom quintile forward return.

Usage: python src/research/factor_ic.py [--db data/research/research.duckdb]
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
H = 21  # forward horizon / sampling step, trading days


def load_prices(db: str) -> pd.DataFrame:
    cfg = yaml.safe_load(open(ROOT / "src" / "config" / "config.yaml"))
    bench = set(cfg.get("benchmark_tickers", []))
    df = duckdb.connect(db, read_only=True).execute(
        "SELECT CAST(date AS DATE) d, stock_id, close FROM price_daily_full"
    ).fetchdf()
    df = df[~df.stock_id.isin(bench)]
    return df.pivot(index="d", columns="stock_id", values="close").sort_index()


def factors(px: pd.DataFrame) -> dict:
    return {
        "mom_12_1": px.shift(21) / px.shift(252) - 1,   # skip last month
        "mom_12":   px / px.shift(252) - 1,             # no skip, for comparison
        "rev_1m":   -(px / px.shift(21) - 1),           # short-term reversal
    }


def ic_table(factor: pd.DataFrame, px: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    fwd = px.shift(-H) / px - 1
    rows = []
    for t in factor.index[252::H]:
        f, r = factor.loc[t], fwd.loc[t]
        ok = f.notna() & r.notna()
        if ok.sum() < min_n:
            continue
        f, r = f[ok], r[ok]
        q = pd.qcut(f.rank(method="first"), 5, labels=False)
        rows.append((t, f.rank().corr(r.rank()),
                     r[q == 4].mean() - r[q == 0].mean(), int(ok.sum())))
    return pd.DataFrame(rows, columns=["date", "ic", "q5_minus_q1", "n"]).set_index("date")


def summarize(t: pd.DataFrame) -> dict:
    n = len(t)
    return {"n_dates": n, "mean_ic": t.ic.mean(), "std_ic": t.ic.std(),
            "t_stat": t.ic.mean() / (t.ic.std() / np.sqrt(n)),
            "hit_rate": (t.ic > 0).mean(), "q5_minus_q1_mo": t.q5_minus_q1.mean()}


def _selfcheck() -> None:
    # factor == forward return (perfect foresight) must give IC ~ 1
    rng = np.random.default_rng(0)
    px = pd.DataFrame(100 * np.exp(rng.normal(0, .02, (600, 40)).cumsum(0)),
                      index=pd.bdate_range("2020-01-01", periods=600))
    t = ic_table(px.shift(-H) / px - 1, px)
    assert t.ic.mean() > 0.99, t.ic.mean()
    t = ic_table(pd.DataFrame(rng.normal(size=px.shape), index=px.index, columns=px.columns), px)
    assert abs(t.ic.mean()) < 0.1, t.ic.mean()


if __name__ == "__main__":
    _selfcheck()
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data/research/research.duckdb"))
    px = load_prices(ap.parse_args().db)
    print(f"{px.shape[1]} stocks, {px.index.min()} -> {px.index.max()}, step={H}d\n")
    res = {}
    for name, f in factors(px).items():
        tab = ic_table(f, px)
        res[name] = summarize(tab)
        if name == "mom_12_1":
            print("mom_12_1 mean IC by year:\n", tab.ic.groupby(pd.to_datetime(tab.index).year).mean().round(3).to_string(), "\n")
    print(pd.DataFrame(res).T.round(4).to_string())
