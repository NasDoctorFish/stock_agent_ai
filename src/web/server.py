"""
Stock_ML Web Dashboard — Tornado Server
========================================
Serves a Vue 3 SPA and exposes REST API endpoints for real-time data.

Usage:
    python src/web/server.py              # default port 8080
    python src/web/server.py --port 9090  # custom port

Endpoints:
    GET /                          — Vue SPA
    GET /api/status                — System heartbeat + last run timestamp
    GET /api/regime                — Latest US + KR regime data (cached 5 min)
    GET /api/valuation             — Market valuation (P/E, Fed Model)
    GET /api/stocks/us             — US top stock picks
    GET /api/stocks/kr             — KR top stock picks with Kiwoom codes
    GET /api/crypto                — Live Coinone tickers (BTC, ETH, XRP, SOL)
    GET /api/orders                — Most recent order log
    GET /api/reports               — List of saved report files
    GET /api/report/<filename>     — Raw Markdown of a specific report

All /api/ responses return JSON with Content-Type: application/json.
CORS is allowed for localhost development.

The server runs alongside the existing terminal logging — StatusLogger writes
to the console/file as before; Tornado uses Python's stdlib logging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import tornado.ioloop
import tornado.web
import tornado.escape
from dotenv import load_dotenv

# Ensure src/ on path
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

load_dotenv()

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
STATIC_DIR  = Path(__file__).resolve().parent / "static"

# ── Simple TTL cache ──────────────────────────────────────────────────────────

class _TTLCache:
    """Tiny in-process TTL cache (no external deps)."""

    def __init__(self):
        self._store: Dict[str, tuple[Any, float]] = {}

    def get(self, key: str, ttl: float) -> Optional[Any]:
        entry = self._store.get(key)
        if entry and (time.time() - entry[1]) < ttl:
            return entry[0]
        return None

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.time())


_cache = _TTLCache()

# ── Data helpers ───────────────────────────────────────────────────────────────

def _regime_data() -> Dict[str, Any]:
    """Run regime detectors (cached 5 minutes)."""
    cached = _cache.get("regime", ttl=300)
    if cached:
        return cached

    try:
        fred_key = os.getenv("FRED_API_KEY", "")
        from regime_detector.us_regime import USRegimeDetector
        from regime_detector.kr_regime import KRRegimeDetector

        us = USRegimeDetector(fred_api_key=fred_key or None).detect()
        kr = KRRegimeDetector(fred_api_key=fred_key or None).detect()
        result = {"us": us, "kr": kr, "fetched_at": datetime.utcnow().isoformat()}
    except Exception as exc:
        logger.warning(f"[Web] Regime detection failed: {exc}")
        result = {"us": {}, "kr": {}, "error": str(exc), "fetched_at": datetime.utcnow().isoformat()}

    _cache.set("regime", result)
    return result


def _valuation_data() -> Dict[str, Any]:
    """Run market timer (cached 5 minutes)."""
    cached = _cache.get("valuation", ttl=300)
    if cached:
        return cached

    try:
        from market_timer.market_timer import MarketTimer
        result = MarketTimer(fred_api_key=os.getenv("FRED_API_KEY") or None).run()
    except Exception as exc:
        logger.warning(f"[Web] Valuation failed: {exc}")
        result = {"us": {}, "kr": {}, "error": str(exc)}

    _cache.set("valuation", result)
    return result


def _kr_stocks_data() -> Dict[str, Any]:
    """Run KR analyst + decision maker (cached 10 minutes)."""
    cached = _cache.get("kr_stocks", ttl=600)
    if cached:
        return cached

    try:
        from core.status_logger import StatusLogger
        from analysts.kr_stock_analyst import KRStockDataAnalyst
        from core.investor import Investor
        from core.kr_decision_maker import KRDecisionMaker

        log = StatusLogger("Web-KR")
        analyst = KRStockDataAnalyst(logger_=log)
        investor = Investor([analyst], log)
        dm = KRDecisionMaker(investor, logger_=log)
        result = dm.run()
        result["fetched_at"] = datetime.utcnow().isoformat()
    except Exception as exc:
        logger.warning(f"[Web] KR stocks failed: {exc}")
        result = {"top_stocks": [], "error": str(exc), "fetched_at": datetime.utcnow().isoformat()}

    _cache.set("kr_stocks", result)
    return result


def _crypto_data() -> Dict[str, Any]:
    """Fetch live crypto tickers from Coinone (cached 1 minute)."""
    cached = _cache.get("crypto", ttl=60)
    if cached:
        return cached

    tickers_out = {}
    try:
        from brokers.coinone_broker import CoinoneBroker
        broker = CoinoneBroker(paper_trading=True)
        for sym in ["BTC", "ETH", "XRP", "SOL", "ADA", "DOGE"]:
            try:
                tickers_out[sym] = broker.get_ticker(sym)
            except Exception:
                tickers_out[sym] = {"currency": sym, "error": "fetch failed"}
    except Exception as exc:
        logger.warning(f"[Web] Crypto fetch failed: {exc}")

    result = {"tickers": tickers_out, "fetched_at": datetime.utcnow().isoformat()}
    _cache.set("crypto", result)
    return result


def _latest_report_data() -> Dict[str, Any]:
    """Parse the most recent saved report file."""
    cached = _cache.get("latest_report", ttl=120)
    if cached:
        return cached

    try:
        reports = sorted(REPORTS_DIR.glob("report_*.md"), reverse=True)
        if reports:
            text = reports[0].read_text(encoding="utf-8")
            result = {"filename": reports[0].name, "content": text, "fetched_at": datetime.utcnow().isoformat()}
        else:
            result = {"filename": None, "content": None}
    except Exception as exc:
        result = {"error": str(exc)}

    _cache.set("latest_report", result)
    return result


def _order_log_data() -> list:
    """Return orders from the most recent report (simple parse)."""
    report = _latest_report_data()
    content = report.get("content", "") or ""
    orders = []
    in_orders = False
    for line in content.splitlines():
        if "Orders This Cycle" in line:
            in_orders = True
            continue
        if in_orders:
            if line.startswith("##"):
                break
            if line.strip().startswith("-"):
                orders.append(line.strip("- ").strip())
    return orders


# ── Base handler ──────────────────────────────────────────────────────────────

class _BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")
        self.set_header("Access-Control-Allow-Origin", "*")

    def write_json(self, data: Any) -> None:
        self.write(json.dumps(data, ensure_ascii=False, default=str))

    def options(self, *args):
        self.set_status(204)
        self.finish()


# ── API handlers ───────────────────────────────────────────────────────────────

class StatusHandler(_BaseHandler):
    def get(self):
        reports = sorted(REPORTS_DIR.glob("report_*.md"), reverse=True)
        last_report = reports[0].name if reports else None
        self.write_json({
            "status": "running",
            "server_time_utc": datetime.utcnow().isoformat(),
            "last_report": last_report,
            "report_count": len(reports),
        })


class RegimeHandler(_BaseHandler):
    def get(self):
        self.write_json(_regime_data())


class ValuationHandler(_BaseHandler):
    def get(self):
        self.write_json(_valuation_data())


class USStocksHandler(_BaseHandler):
    def get(self):
        report = _latest_report_data()
        content = report.get("content", "") or ""
        # Extract US top stocks section
        lines, in_section = [], False
        for line in content.splitlines():
            if "US Top Stocks" in line:
                in_section = True
                continue
            if in_section:
                if line.startswith("##"):
                    break
                if line.strip():
                    lines.append(line.strip())
        self.write_json({"section": lines, "fetched_at": report.get("fetched_at")})


class KRStocksHandler(_BaseHandler):
    def get(self):
        self.write_json(_kr_stocks_data())


class CryptoHandler(_BaseHandler):
    def get(self):
        self.write_json(_crypto_data())


class OrdersHandler(_BaseHandler):
    def get(self):
        self.write_json({"orders": _order_log_data()})


class ReportsListHandler(_BaseHandler):
    def get(self):
        try:
            files = sorted(REPORTS_DIR.glob("report_*.md"), reverse=True)
            self.write_json({
                "reports": [f.name for f in files[:30]],
                "total": len(files),
            })
        except Exception as exc:
            self.write_json({"reports": [], "error": str(exc)})


class ReportFileHandler(_BaseHandler):
    def get(self, filename: str):
        # Safety: only allow report_*.md filenames
        if not filename.startswith("report_") or not filename.endswith(".md"):
            self.set_status(400)
            self.write_json({"error": "Invalid filename"})
            return
        path = REPORTS_DIR / filename
        if not path.exists():
            self.set_status(404)
            self.write_json({"error": "Not found"})
            return
        self.write_json({"filename": filename, "content": path.read_text(encoding="utf-8")})


class RefreshCacheHandler(_BaseHandler):
    """POST /api/refresh — invalidate all cache entries."""
    def post(self):
        _cache._store.clear()
        self.write_json({"ok": True, "message": "Cache cleared"})


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        index = STATIC_DIR / "index.html"
        if index.exists():
            self.set_header("Content-Type", "text/html")
            self.write(index.read_text(encoding="utf-8"))
        else:
            self.write("<h1>Stock_ML Dashboard</h1><p>index.html not found.</p>")


# ── App factory ────────────────────────────────────────────────────────────────

def make_app() -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/",                           IndexHandler),
            (r"/api/status",                 StatusHandler),
            (r"/api/regime",                 RegimeHandler),
            (r"/api/valuation",              ValuationHandler),
            (r"/api/stocks/us",              USStocksHandler),
            (r"/api/stocks/kr",              KRStocksHandler),
            (r"/api/crypto",                 CryptoHandler),
            (r"/api/orders",                 OrdersHandler),
            (r"/api/reports",                ReportsListHandler),
            (r"/api/report/([^/]+\.md)",     ReportFileHandler),
            (r"/api/refresh",                RefreshCacheHandler),
            (r"/static/(.*)",                tornado.web.StaticFileHandler,
             {"path": str(STATIC_DIR)}),
        ],
        debug=False,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Stock_ML Web Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default 8080)")
    args = parser.parse_args()

    app = make_app()
    app.listen(args.port)
    logger.info(f"[Web] Stock_ML dashboard running at http://localhost:{args.port}")
    logger.info(f"[Web] Static dir : {STATIC_DIR}")
    logger.info(f"[Web] Reports dir: {REPORTS_DIR}")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
