"""Run: STOCKML_DB=<db copy> poetry run pytest src/stock_ml/tests -q   (no real Anthropic call is made)."""
import os
import shutil
import time
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from stock_ml.backend import ai, app as appmod, jobs
from stock_ml.backend.digest import build_digest, digest_json
from stock_ml.backend.schemas import Strategy

pytestmark = pytest.mark.skipif(not os.environ.get("STOCKML_DB"), reason="set STOCKML_DB to a DB copy")

STRAT = {"name": "Mock", "rationale": "r", "top_n": 5, "rebalance": "monthly",
         "equity_by_regime": {"risk_on": 1, "cautious": 0.6, "defensive": 0.3, "risk_off": 0},
         "sell_rules": {"stop_loss_pct": 8, "max_holding_days": 60}, "rebalance_triggers": [], "risk_notes": "n"}
PARAMS = {"start": "2026-03-02", "end": "2026-09-30", "rebalance": "monthly", "top_n": 5, "capital": 140_000_000,
          "currency": "KRW", "fx": 1400, "benchmark": "SPY", "slippage_pct": 0.1, "commission_pct": 0.015}


class FakeClient:
    def __init__(self, delay=0):
        self.calls, self.delay = [], delay
        self.messages = NS(create=self.create)

    def create(self, **kw):
        time.sleep(self.delay)
        self.calls.append(kw)
        assert kw["max_tokens"] <= 2000
        if "tools" in kw:
            return NS(content=[NS(type="tool_use", input=STRAT)])
        return NS(content=[NS(type="text", text="## Review\nok")])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    shutil.copytree(appmod.STRAT_DIR, tmp_path / "strategies")
    monkeypatch.setattr(appmod, "STRAT_DIR", tmp_path / "strategies")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    fake = FakeClient()
    monkeypatch.setattr(ai, "get_client", lambda: fake)
    c = TestClient(appmod.app)
    c.fake = fake
    return c


def wait(c, jid, until=("done", "error", "cancelled"), t=120):
    end = time.time() + t
    while time.time() < end:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in until:
            return j
        time.sleep(0.3)
    raise TimeoutError


def test_digest_small():
    d = build_digest()
    assert len(digest_json(d)) < 10_000 and d["regime"]["label"] and d["macro"] and d["stocks"]["top_momentum"]


def test_schema():
    assert Strategy.model_validate(STRAT).top_n == 5
    with pytest.raises(ValidationError):
        Strategy.model_validate({**STRAT, "equity_by_regime": {"risk_on": 1.5}})
    with pytest.raises(ValidationError):
        Strategy.model_validate({**STRAT, "rebalance": "daily"})


def test_config_and_templates(client):
    cfg = client.get("/api/config").json()
    assert cfg["ai_enabled"] and "sk-test" not in str(cfg) and cfg["data_end"] and "SPY" in cfg["benchmarks"]
    assert len(client.get("/api/strategies").json()) >= 5
    saved = client.post("/api/strategies", json=STRAT).json()
    assert saved["id"].startswith("user-")


def test_backtest_and_strategy_changes_result(client):
    base = client.post("/api/backtest", json=PARAMS).json()
    assert base["nav"] and base["drawdown"] and base["regimes"] and base["stats"]["total_return"] is not None
    assert base["bench_stats"] and base["nav"][0]["benchmark"]
    strat = client.post("/api/backtest", json={**PARAMS, "strategy": {**STRAT, "equity_by_regime": {
        "risk_on": 0.2, "cautious": 0.2, "defensive": 0.2, "risk_off": 0.2}}}).json()
    assert strat["nav"][-1]["strategy"] != base["nav"][-1]["strategy"]


def test_job_lifecycle_and_events(client):
    jid = client.post("/api/analyze", json={"mode": "full_ai", "params": PARAMS}).json()["job_id"]
    assert any(j["id"] == jid for j in client.get("/api/jobs").json())
    j = wait(client, jid)
    assert j["status"] == "done", j
    r = j["result"]
    assert r["strategy"]["name"] == "Mock" and r["backtest"]["nav"] and "Review" in r["report_markdown"]
    assert r["digest_bytes"] < 10_000 and len(client.fake.calls) == 2
    assert '"done"' in client.get(f"/api/jobs/{jid}/events").text


def test_job_cancel(client, monkeypatch):
    monkeypatch.setattr(ai, "get_client", lambda: FakeClient(delay=3))
    jid = client.post("/api/analyze", json={"mode": "recommend", "params": PARAMS}).json()["job_id"]
    time.sleep(1)
    assert client.post(f"/api/jobs/{jid}/cancel").json()["status"] == "cancelled"
    time.sleep(5)
    assert client.get(f"/api/jobs/{jid}").json()["status"] == "cancelled"


def test_analyze_requires_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert client.post("/api/analyze", json={"mode": "recommend", "params": PARAMS}).status_code == 503
