from datetime import datetime, timezone

import pandas as pd

from plunge.detector import evaluate, market_open


def test_evaluate_threshold_and_reference_excludes_today():
    c = pd.Series([100, 110, 105, 104.0])
    a = evaluate("X", c, 30, 5.0)           # 110 -> 104 = 5.45%
    assert a and a["reference_price"] == 110 and a["drop_pct"] == 5.45
    assert evaluate("X", c, 30, 6.0) is None
    assert evaluate("X", pd.Series([100.0, 120.0]), 30, 5.0) is None  # new high is not a plunge


def test_market_hours():
    mon_1500utc = datetime(2026, 10, 26, 15, 0, tzinfo=timezone.utc)  # 11:00 NY (EDT), 00:00 Tue KST
    assert market_open("AAPL", mon_1500utc) and not market_open("005930.KS", mon_1500utc)
    assert not market_open("AAPL", datetime(2026, 10, 24, 15, 0, tzinfo=timezone.utc))  # Saturday
