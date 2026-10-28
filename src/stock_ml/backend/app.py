"""FastAPI app. Run: see run_stock_ml.sh. Serves ../frontend as static at '/'."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import ai, jobs, service
from .digest import build_digest, digest_json
from .schemas import AnalyzeRequest, BacktestParams, Strategy

STRAT_DIR = Path(__file__).resolve().parents[1] / "strategies"
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="Stock_ML")
jobs.mark_interrupted()


@app.get("/api/config")
def config():
    m = service.db_meta()
    return {"ai_enabled": ai.ai_enabled(), "models": ai.MODELS, "default_model": ai.DEFAULT_MODEL,
            "currencies": service.CURRENCIES, "benchmarks": m["benchmarks"], "data_end": m["data_end"],
            "data_max": m["data_max"], "tickers": m["tickers"]}


@app.post("/api/backtest")
def backtest(p: BacktestParams):
    try:
        return service.run_backtest(p)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"backtest failed: {e}")


# ── strategies ──
def _load_strategies():
    out = []
    for f in sorted(STRAT_DIR.glob("*.json")):
        try:
            out.append(Strategy.model_validate_json(f.read_text()).model_dump())
        except Exception:
            continue
    return out


@app.get("/api/strategies")
def strategies():
    return _load_strategies()


@app.post("/api/strategies")
def save_strategy(s: Strategy):
    sid = re.sub(r"[^a-z0-9_-]+", "-", (s.id or s.name).lower()).strip("-")[:60] or "strategy"
    if not (STRAT_DIR / f"{sid}.json").exists() or not sid.startswith("user-"):
        sid = "user-" + sid.removeprefix("user-")  # never overwrite templates
    s.id = sid
    STRAT_DIR.mkdir(exist_ok=True)
    (STRAT_DIR / f"{sid}.json").write_text(s.model_dump_json(indent=2))
    return s.model_dump()


# ── analysis jobs ──
def _analysis(req: AnalyzeRequest):
    def work(jid: str) -> dict:
        step = lambda pct: (jobs.check(jid), jobs.update(jid, progress=pct))  # noqa: E731
        params = req.params
        digest = build_digest(as_of=None if not params.end else min(params.end, service.db_meta()["data_end"]))
        step(20)
        full = req.mode == "full_ai"
        strat = ai.propose_strategy(digest, params.model_dump(exclude={"strategy"}), req.notes, req.model, full=full)
        step(50)
        p = params.model_copy(update={"strategy": strat})
        if full:  # AI also picks top_n / rebalance schedule
            p = p.model_copy(update={"top_n": strat.top_n, "rebalance": strat.rebalance,
                                     "rebalance_days": strat.rebalance_days})
        bt = service.run_backtest(p)
        step(75)
        if full:
            report = ai.review(build_digest(as_of=digest["as_of"], backtest=bt), strat, req.model)
        else:
            report = f"## {strat.name}\n\n{strat.rationale}\n\n**Risks:** {strat.risk_notes}"
        return {"strategy": strat.model_dump(), "report_markdown": report, "backtest": bt,
                "digest_bytes": len(digest_json(digest))}
    return work


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if not ai.ai_enabled():
        raise HTTPException(503, "AI disabled: set ANTHROPIC_API_KEY (env or .env)")
    title = f"{req.mode} {req.params.start}..{req.params.end}"
    return {"job_id": jobs.submit("analyze", title, _analysis(req))}


@app.get("/api/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "no such job")
    return j


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    j = jobs.cancel(jid)
    if not j:
        raise HTTPException(404, "no such job")
    return j


@app.get("/api/jobs/{jid}/events")
def job_events(jid: str):
    if not jobs.get(jid):
        raise HTTPException(404, "no such job")
    return StreamingResponse(jobs.events(jid), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── plunge detector (optional feature) ──
@app.get("/api/plunge/status")
def plunge_status():
    from plunge import detector
    return {"alerts": detector.get_alerts(20), "watchlist": list(detector.get_status().values())}


@app.post("/api/plunge/scan")
def plunge_scan():
    from plunge import detector
    try:
        return {"new_alerts": detector.scan()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"scan failed: {e}")


if FRONTEND.exists():  # mounted last so /api/* wins
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
