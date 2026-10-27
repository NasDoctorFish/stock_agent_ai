"""
Price-familiarity and relative-valuation features (point-in-time).

Price features at date t use prices <= t only. Valuation uses quarterly
diluted EPS; the DB has no filing dates, so EPS of fiscal quarter F is
usable only from F + EPS_LAG_DAYS (calendar). No book equity exists in
fundamentals_q_full, so P/B is not computed.

Usage (self-check): poetry run python src/research/familiarity.py
"""
import duckdb
import numpy as np
import pandas as pd

EPS_LAG_DAYS = 45
WINDOWS = (63, 126, 252)


def price_familiarity(px: pd.DataFrame) -> dict:
    lp = np.log(px)
    f = {}
    for w in WINDOWS:
        f[f"z_{w}"] = (lp - lp.rolling(w).mean()) / lp.rolling(w).std()
    for w in (252, 756):
        # percentile of today's price within the trailing window (incl. today)
        f[f"pctl_{w}"] = px.rolling(w).apply(lambda a: (a <= a[-1]).mean(), raw=True)
    hi, lo = px.rolling(252).max(), px.rolling(252).min()
    f["range_252"] = (px - lo) / (hi - lo)
    f["dd_126"] = px / px.rolling(126).max() - 1
    f["ma_dist_200"] = px / px.rolling(200).mean() - 1
    f["ma_dist_500"] = px / px.rolling(500).mean() - 1
    return f


def load_eps(db: str) -> pd.DataFrame:
    q = duckdb.connect(db, read_only=True).execute(
        'SELECT stock_id, CAST(fiscal_date AS DATE) fd, "Diluted EPS" eps '
        "FROM fundamentals_q_full WHERE fiscal_date IS NOT NULL AND \"Diluted EPS\" IS NOT NULL"
    ).fetchdf()
    return q.drop_duplicates(["stock_id", "fd"]).sort_values(["stock_id", "fd"])


def ttm_eps(eps: pd.DataFrame, index: pd.DatetimeIndex, columns, lag: int = EPS_LAG_DAYS) -> pd.DataFrame:
    """TTM EPS per (date, stock); needs 4 consecutive-ish quarters, known only after fd+lag."""
    out = pd.DataFrame(np.nan, index=index, columns=columns)
    for s, g in eps.groupby("stock_id"):
        if s not in out.columns:
            continue
        g = g.set_index("fd")["eps"]
        t = g.rolling(4).sum()
        # reject TTM spanning >400 days (missing quarter)
        t[(g.index.to_series().diff(3).dt.days > 400).values] = np.nan
        avail = pd.Series(t.values, index=pd.to_datetime(t.index) + pd.Timedelta(days=lag)).dropna()
        if avail.empty:
            continue
        out[s] = avail.reindex(index.union(avail.index)).ffill().reindex(index)
    return out


def valuation(px: pd.DataFrame, eps: pd.DataFrame, lag: int = EPS_LAG_DAYS) -> dict:
    e = ttm_eps(eps, px.index, px.columns, lag)
    ey = e / px
    mu, sd = ey.expanding(63).mean(), ey.expanding(63).std()
    return {
        "ey": ey,
        "pe_inv_rel": (ey - mu) / sd,                       # z of earnings yield vs own history
        "ey_pctl": ey.expanding(63).apply(lambda a: (a <= a[-1]).sum() / np.isfinite(a).sum(), raw=True),
        "pe": (px / e).where(e > 0),
    }


def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=1200)
    px = pd.DataFrame(100 * np.exp(rng.normal(0, .02, (1200, 5)).cumsum(0)), index=idx, columns=list("ABCDE"))
    cut = 900
    full, part = price_familiarity(px), price_familiarity(px.iloc[:cut])
    for k in full:
        a, b = full[k].iloc[:cut], part[k]
        assert np.allclose(a.values, b.values, equal_nan=True), f"look-ahead in {k}"
    fd = pd.to_datetime(["2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31", "2019-03-31"])
    eps = pd.DataFrame({"stock_id": "A", "fd": fd, "eps": [1., 1, 1, 1, 5]})
    e = ttm_eps(eps, idx, px.columns)["A"]
    assert e[: pd.Timestamp("2019-02-13")].isna().all(), "EPS used before lag"
    assert e[pd.Timestamp("2019-02-14")] == 4.0
    assert e[pd.Timestamp("2019-05-15")] == 8.0
    print("familiarity self-checks ok")


if __name__ == "__main__":
    _selfcheck()
