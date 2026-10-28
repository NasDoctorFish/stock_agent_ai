import numpy as np
import pandas as pd

from core.scoring import LOOKBACK, SKIP, mom_12_1, score_prices


def _px(n=300):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"UP": np.linspace(100, 200, n), "FLAT": 100.0, "DOWN": np.linspace(100, 50, n)}, index=idx)


def test_formula_and_skip_month():
    px = _px()
    m = mom_12_1(px)
    assert np.isclose(m["UP"], px["UP"].iloc[-1 - SKIP] / px["UP"].iloc[-1 - LOOKBACK] - 1)
    px.iloc[-1] *= 10  # last-day price must not change a skip-month score
    assert np.allclose(mom_12_1(px), m)


def test_ranking_and_short_history():
    s = score_prices(_px())
    assert s["UP"] > s["FLAT"] > s["DOWN"] and abs(s.mean()) < 1e-9
    assert score_prices(_px(100)).empty

