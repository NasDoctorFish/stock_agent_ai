import numpy as np
import pandas as pd

from pitdata import manifest, pit, quality


def _q():
    fd = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31", "2021-03-31"])
    return pd.DataFrame({"stock_id": "A", "fd": fd, "eps": [1., 1, 1, 1, 5]})


def test_not_visible_before_lag():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    for lag in pit.LAGS:
        e = pit.ttm_panel(_q(), "eps", idx, ["A"], lag)["A"]
        first = pd.Timestamp("2020-12-31") + pd.Timedelta(days=lag)
        assert e[e.index < first].isna().all()
        assert e[e.index >= first].iloc[0] == 4.0
        nxt = pd.Timestamp("2021-03-31") + pd.Timedelta(days=lag)
        assert e[e.index >= nxt].iloc[0] == 8.0 and e[e.index < nxt].dropna().eq(4.0).all()


def test_gap_gives_nan_not_fill():
    q = _q().drop(index=1)  # missing quarter -> TTM spans >400d? 3 quarters only before 2021 => NaN
    e = pit.ttm_panel(q, "eps", pd.bdate_range("2020-01-01", "2021-12-31"), ["A"], 45)["A"]
    assert e[:"2021-03-01"].isna().all()


def test_manifest_deterministic():
    px = pd.DataFrame(np.arange(20.).reshape(10, 2) + 1, index=pd.bdate_range("2020-01-01", periods=10), columns=["A", "B"])
    q = _q()
    assert manifest.build(px, q, 45)["version"] == manifest.build(px, q, 45)["version"]
    assert manifest.build(px, q, 45)["version"] != manifest.build(px, q, 60)["version"]


def test_quality_flags_jump():
    px = pd.DataFrame({"A": [1., 1, 3, 3]}, index=pd.bdate_range("2020-01-01", periods=4))
    assert quality.price_report(px)["jumps_gt"]["n"] == 1


def test_real_data_smoke():
    px, q = pit.load_universe(), pit.load_quarterly()
    assert quality.price_report(px)["nonpositive"] == 0
    assert quality.fundamentals_report(q)["dup_keys"] == 0
