"""In-process job store: thread pool + one JSON file per job (survives restarts/page refresh)."""
from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

JOBS_DIR = Path(__file__).resolve().parents[1] / "jobs"
TERMINAL = {"done", "error", "cancelled"}

_pool = ThreadPoolExecutor(max_workers=2)
_lock = threading.RLock()
_cancel: dict[str, threading.Event] = {}


class Cancelled(Exception):
    pass


def _path(jid: str) -> Path:
    return JOBS_DIR / f"{Path(jid).name}.json"


def _save(job: dict) -> None:
    JOBS_DIR.mkdir(exist_ok=True)
    with _lock:
        tmp = _path(job["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(job))
        tmp.replace(_path(job["id"]))


def get(jid: str) -> Optional[dict]:
    try:
        return json.loads(_path(jid).read_text())
    except Exception:
        return None


def list_jobs() -> list[dict]:
    JOBS_DIR.mkdir(exist_ok=True)
    jobs = [j for j in (get(p.stem) for p in JOBS_DIR.glob("*.json")) if j]
    return sorted(({k: j.get(k) for k in ("id", "kind", "status", "created", "progress", "title")} for j in jobs),
                  key=lambda j: j["created"], reverse=True)


def mark_interrupted() -> None:
    """Jobs left queued/running by a previous process can never finish."""
    for j in list_jobs():
        if j["status"] not in TERMINAL:
            full = get(j["id"])
            full.update(status="error", error="interrupted by server restart")
            _save(full)


def update(jid: str, **kw) -> None:
    with _lock:
        j = get(jid)
        if j and j["status"] not in TERMINAL:
            j.update(kw)
            _save(j)


def check(jid: str) -> None:
    if _cancel[jid].is_set():
        raise Cancelled()


def submit(kind: str, title: str, fn: Callable[[str], dict]) -> str:
    """fn(job_id) -> result dict; call jobs.check(id)/update(id, progress=..) inside."""
    jid = uuid.uuid4().hex[:12]
    _cancel[jid] = threading.Event()
    _save({"id": jid, "kind": kind, "title": title, "status": "queued", "progress": 0,
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "result": None, "error": None})

    def run():
        try:
            check(jid)
            update(jid, status="running", progress=5)
            res = fn(jid)
            check(jid)
            update(jid, status="done", progress=100, result=res)
        except Cancelled:
            pass  # cancel() already wrote status
        except Exception as e:  # noqa: BLE001
            update(jid, status="error", error=str(e)[:500])
    _pool.submit(run)
    return jid


def cancel(jid: str) -> Optional[dict]:
    with _lock:
        j = get(jid)
        if not j:
            return None
        if j["status"] not in TERMINAL:
            _cancel.setdefault(jid, threading.Event()).set()
            j["status"] = "cancelled"
            _save(j)
        return j


def events(jid: str):
    """SSE generator: emits on status/progress change, ends on terminal state."""
    last = None
    while True:
        j = get(jid)
        if not j:
            yield "event: error\ndata: {}\n\n"
            return
        cur = (j["status"], j["progress"])
        if cur != last:
            last = cur
            yield f"data: {json.dumps({k: j[k] for k in ('id', 'status', 'progress', 'error')})}\n\n"
        if j["status"] in TERMINAL:
            return
        time.sleep(0.5)
