"""
24/7 Scheduler
==============
Runs the full US + KR analysis pipeline on a recurring schedule using
Python's APScheduler library.  Falls back to a simple threading-based
loop if APScheduler is not installed.

Job schedule (all times in UTC):
  ┌──────────────────────────────────────────────────────────────┐
  │ Job                  │ Interval │ Notes                       │
  ├──────────────────────────────────────────────────────────────┤
  │ data_update          │ 30 min   │ Refresh CSV / DuckDB        │
  │ us_analysis          │ 60 min   │ Full US analyst pipeline    │
  │ kr_analysis          │ 60 min   │ Full KR analyst pipeline    │
  │ market_regime        │ 60 min   │ US + KR regime detectors    │
  │ market_timing        │ 60 min   │ Valuation / market timer    │
  │ send_report          │  3 hr    │ Telegram 3-hour report      │
  │ daily_report         │ 2x/day   │ 04:00 & 16:00 UTC            │
  │                      │          │ (00:00 & 12:00 SGT)          │
  └──────────────────────────────────────────────────────────────┘

The scheduler runs an initial pass of all jobs on startup before
entering the recurring schedule.

Usage:
    python -m scheduler.scheduler
    # or via main.py --daemon

Graceful shutdown:
    SIGINT / SIGTERM → scheduler stops cleanly
    Errors in individual jobs are caught and logged; the scheduler
    continues running unless _MAX_CONSECUTIVE_ERRORS is exceeded.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# Ensure src/ is on sys.path when run directly
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger(__name__)

# How many consecutive job failures before aborting the entire scheduler
_MAX_CONSECUTIVE_ERRORS = 10

# ── APScheduler detection ──────────────────────────────────────────────────────
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger
    _HAS_APSCHEDULER = True
except ImportError:
    _HAS_APSCHEDULER = False
    logger.warning(
        "APScheduler not installed. Install with: pip install apscheduler\n"
        "Falling back to a simple sleep-loop scheduler."
    )


# ── Pipeline components ────────────────────────────────────────────────────────

class SchedulerPipeline:
    """
    Holds all pipeline state shared across scheduler jobs.
    Initialized once at startup; jobs read/write to this shared store.
    """

    def __init__(self):
        # Import here so individual import errors don't crash the whole process
        from core.status_logger import StatusLogger
        from core.db_handler import DBHandler
        from core.db_updater import DBUpdater
        from core.investor import Investor
        from core.decision_maker import DecisionMaker

        from analysts.macro_analyst import MacroAnalyst
        from analysts.fundamental_analyst import FundamentalAnalyst
        from analysts.stock_data_analyst import StockDataAnalyst
        from analysts.news_media_analyst import NewsMediaAnalyst
        from analysts.business_analyst import BusinessAnalyst
        from analysts.kr_stock_analyst import KRStockDataAnalyst

        from regime_detector.us_regime import USRegimeDetector
        from regime_detector.kr_regime import KRRegimeDetector
        from market_timer.market_timer import MarketTimer
        from reporters.report_generator import ReportGenerator
        from reporters.daily_report import DailyReportGenerator

        self.logger = StatusLogger("scheduler")
        self.db = DBHandler(self.logger)
        self.updater = DBUpdater(self.db, self.logger)

        fred_key = os.getenv("FRED_API_KEY")

        # US analysts
        self.us_analysts = [
            MacroAnalyst(self.db, self.logger),
            FundamentalAnalyst(self.db, self.logger),
            StockDataAnalyst(self.db, self.logger),
            NewsMediaAnalyst(self.db, self.logger),
            BusinessAnalyst(self.db, self.logger),
        ]
        self.us_investor = Investor(self.us_analysts, self.logger)
        self.us_decision_maker = DecisionMaker(self.us_investor, self.logger)

        # KR analysts
        self.kr_stock_analyst = KRStockDataAnalyst(logger_=self.logger)
        self.kr_investor = Investor([self.kr_stock_analyst], self.logger)
        self.kr_decision_maker = DecisionMaker(self.kr_investor, self.logger)

        # Regime detectors
        self.us_regime_detector = USRegimeDetector(fred_api_key=fred_key)
        self.kr_regime_detector = KRRegimeDetector(fred_api_key=fred_key)

        # Market timer
        self.market_timer = MarketTimer(fred_api_key=fred_key)

        # Reporter (sends via Discord webhook, configured from env)
        self.reporter = ReportGenerator()
        self.daily_reporter = DailyReportGenerator()

        # Shared state updated by jobs
        self.last_us_regime: Optional[Dict] = None
        self.last_kr_regime: Optional[Dict] = None
        self.last_market_timing: Optional[Dict] = None
        self.last_us_decision: Optional[Dict] = None
        self.last_kr_decision: Optional[Dict] = None
        self.order_log: List[Dict] = []

        # Error counter
        self.consecutive_errors = 0


# ── Individual job functions ───────────────────────────────────────────────────

def job_data_update(pipeline: SchedulerPipeline) -> None:
    """Refresh all CSV data and reload into DuckDB."""
    pipeline.logger.info("[JOB] data_update — starting")
    try:
        pipeline.updater.update_csv_files()
        pipeline.db.save_all_to_duckdb()
        pipeline.logger.info("[JOB] data_update — complete")
        pipeline.consecutive_errors = 0
    except Exception as exc:
        pipeline.logger.error(f"[JOB] data_update failed: {exc}\n{traceback.format_exc()}")
        pipeline.consecutive_errors += 1


def job_us_analysis(pipeline: SchedulerPipeline) -> None:
    """Run US analyst pipeline and update shared decision state."""
    pipeline.logger.info("[JOB] us_analysis — starting")
    try:
        # Force re-run by clearing cached results on each analyst
        for analyst in pipeline.us_analysts:
            analyst.last_result = None
        pipeline.last_us_decision = pipeline.us_decision_maker.run()
        pipeline.logger.info("[JOB] us_analysis — complete")
        pipeline.consecutive_errors = 0
    except Exception as exc:
        pipeline.logger.error(f"[JOB] us_analysis failed: {exc}\n{traceback.format_exc()}")
        pipeline.consecutive_errors += 1


def job_kr_analysis(pipeline: SchedulerPipeline) -> None:
    """Run KR analyst pipeline and update shared decision state."""
    pipeline.logger.info("[JOB] kr_analysis — starting")
    try:
        pipeline.kr_stock_analyst.last_result = None  # force re-fetch
        pipeline.last_kr_decision = pipeline.kr_decision_maker.run()
        pipeline.logger.info("[JOB] kr_analysis — complete")
        pipeline.consecutive_errors = 0
    except Exception as exc:
        pipeline.logger.error(f"[JOB] kr_analysis failed: {exc}\n{traceback.format_exc()}")
        pipeline.consecutive_errors += 1


def job_market_regime(pipeline: SchedulerPipeline) -> None:
    """Detect US and KR market regimes."""
    pipeline.logger.info("[JOB] market_regime — starting")
    try:
        pipeline.last_us_regime = pipeline.us_regime_detector.detect()
        pipeline.last_kr_regime = pipeline.kr_regime_detector.detect()
        pipeline.logger.info(
            f"[JOB] market_regime — US={pipeline.last_us_regime['regime']} "
            f"KR={pipeline.last_kr_regime['regime']}"
        )
        pipeline.consecutive_errors = 0
    except Exception as exc:
        pipeline.logger.error(f"[JOB] market_regime failed: {exc}\n{traceback.format_exc()}")
        pipeline.consecutive_errors += 1


def job_market_timing(pipeline: SchedulerPipeline) -> None:
    """Run market valuation / timing module."""
    pipeline.logger.info("[JOB] market_timing — starting")
    try:
        pipeline.last_market_timing = pipeline.market_timer.run()
        us_label = pipeline.last_market_timing["us"].get("label", "N/A")
        kr_label = pipeline.last_market_timing["kr"].get("label", "N/A")
        pipeline.logger.info(f"[JOB] market_timing — US={us_label} KR={kr_label}")
        pipeline.consecutive_errors = 0
    except Exception as exc:
        pipeline.logger.error(f"[JOB] market_timing failed: {exc}\n{traceback.format_exc()}")
        pipeline.consecutive_errors += 1


def job_send_report(pipeline: SchedulerPipeline) -> None:
    """Assemble and send the 3-hour status report."""
    pipeline.logger.info("[JOB] send_report — starting")
    try:
        pipeline.reporter.generate_and_send(
            us_regime=pipeline.last_us_regime,
            kr_regime=pipeline.last_kr_regime,
            market_timing=pipeline.last_market_timing,
            us_decision=pipeline.last_us_decision,
            kr_decision=pipeline.last_kr_decision,
            order_log=list(pipeline.order_log),  # snapshot
        )
        # Clear order log after reporting
        pipeline.order_log.clear()
        pipeline.logger.info("[JOB] send_report — complete")
        pipeline.consecutive_errors = 0
    except Exception as exc:
        pipeline.logger.error(f"[JOB] send_report failed: {exc}\n{traceback.format_exc()}")
        pipeline.consecutive_errors += 1


def job_daily_report(pipeline: SchedulerPipeline) -> None:
    """Assemble and send the once-daily market briefing."""
    pipeline.logger.info("[JOB] daily_report — starting")
    try:
        pipeline.daily_reporter.generate_and_send(
            us_regime=pipeline.last_us_regime,
            us_decision=pipeline.last_us_decision,
            kr_decision=pipeline.last_kr_decision,
        )
        pipeline.logger.info("[JOB] daily_report — complete")
        pipeline.consecutive_errors = 0
    except Exception as exc:
        pipeline.logger.error(f"[JOB] daily_report failed: {exc}\n{traceback.format_exc()}")
        pipeline.consecutive_errors += 1


def _run_all_jobs(pipeline: SchedulerPipeline) -> None:
    """Execute all jobs sequentially for the initial warm-up pass."""
    for fn in [
        job_data_update,
        job_market_regime,
        job_market_timing,
        job_us_analysis,
        job_kr_analysis,
        job_send_report,
    ]:
        fn(pipeline)


# ── APScheduler-based implementation ──────────────────────────────────────────

def run_with_apscheduler(pipeline: SchedulerPipeline) -> None:
    """Start the recurring schedule using APScheduler."""
    sched = BackgroundScheduler(timezone="UTC")

    # Register all jobs
    sched.add_job(
        job_data_update, IntervalTrigger(minutes=30),
        args=[pipeline], id="data_update", name="Data Update",
        max_instances=1, coalesce=True,
    )
    sched.add_job(
        job_market_regime, IntervalTrigger(minutes=60),
        args=[pipeline], id="market_regime", name="Market Regime",
        max_instances=1, coalesce=True,
    )
    sched.add_job(
        job_market_timing, IntervalTrigger(minutes=60),
        args=[pipeline], id="market_timing", name="Market Timing",
        max_instances=1, coalesce=True,
    )
    sched.add_job(
        job_us_analysis, IntervalTrigger(minutes=60),
        args=[pipeline], id="us_analysis", name="US Analysis",
        max_instances=1, coalesce=True,
    )
    sched.add_job(
        job_kr_analysis, IntervalTrigger(minutes=60),
        args=[pipeline], id="kr_analysis", name="KR Analysis",
        max_instances=1, coalesce=True,
    )
    sched.add_job(
        job_send_report, IntervalTrigger(hours=3),
        args=[pipeline], id="send_report", name="3h Report",
        max_instances=1, coalesce=True,
    )
    sched.add_job(
        job_daily_report, CronTrigger(hour="4,16", minute=0, timezone="UTC"),
        args=[pipeline], id="daily_report", name="Daily Market Report",
        max_instances=1, coalesce=True,
    )

    sched.start()
    pipeline.logger.info("APScheduler started — all jobs scheduled")
    pipeline.logger.info("  data_update  : every 30 min")
    pipeline.logger.info("  regime/timing: every 60 min")
    pipeline.logger.info("  us/kr_analysis: every 60 min")
    pipeline.logger.info("  send_report  : every 3 hours")
    pipeline.logger.info("  daily_report : 04:00 & 16:00 UTC (00:00 & 12:00 SGT)")

    # Keep the main thread alive; handle graceful shutdown
    stop_event = _setup_signal_handlers()
    try:
        while not stop_event.is_set():
            if pipeline.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                pipeline.logger.error(
                    f"Too many consecutive errors ({_MAX_CONSECUTIVE_ERRORS}). "
                    "Shutting down scheduler."
                )
                break
            time.sleep(5)
    finally:
        pipeline.logger.info("Shutting down scheduler...")
        # 내부 thread 전부 종료될 때까지 기다림, APScheduler 공식 방식
        sched.shutdown(wait=True)
        pipeline.logger.info("Scheduler stopped cleanly")
        sys.exit(0)
    


# ── Simple sleep-loop fallback ────────────────────────────────────────────────

def run_with_sleep_loop(pipeline: SchedulerPipeline) -> None:
    """
    Simple scheduler fallback (no APScheduler dependency).
    Tracks last-run times and executes jobs when their interval has elapsed.
    """
    import threading

    INTERVALS = {
        "data_update":   30 * 60,
        "regime_timing": 60 * 60,
        "us_analysis":   60 * 60,
        "kr_analysis":   60 * 60,
        "send_report":    3 * 3600,
    }

    last_run: Dict[str, float] = {k: 0.0 for k in INTERVALS}
    last_daily_report_slot: Optional[str] = None  # "YYYY-MM-DD-HH", one entry per fired slot
    DAILY_REPORT_HOURS_UTC = (4, 16)  # 00:00 & 12:00 SGT
    stop_event = _setup_signal_handlers()

    pipeline.logger.info("Sleep-loop scheduler started (APScheduler not installed)")

    while not stop_event.is_set():
        now = time.time()

        if now - last_run["data_update"] >= INTERVALS["data_update"]:
            threading.Thread(target=job_data_update, args=(pipeline,), daemon=True).start()
            last_run["data_update"] = now

        if now - last_run["regime_timing"] >= INTERVALS["regime_timing"]:
            threading.Thread(target=job_market_regime, args=(pipeline,), daemon=True).start()
            threading.Thread(target=job_market_timing, args=(pipeline,), daemon=True).start()
            last_run["regime_timing"] = now

        if now - last_run["us_analysis"] >= INTERVALS["us_analysis"]:
            threading.Thread(target=job_us_analysis, args=(pipeline,), daemon=True).start()
            last_run["us_analysis"] = now

        if now - last_run["kr_analysis"] >= INTERVALS["kr_analysis"]:
            threading.Thread(target=job_kr_analysis, args=(pipeline,), daemon=True).start()
            last_run["kr_analysis"] = now

        if now - last_run["send_report"] >= INTERVALS["send_report"]:
            threading.Thread(target=job_send_report, args=(pipeline,), daemon=True).start()
            last_run["send_report"] = now

        now_utc = datetime.utcnow()
        slot = now_utc.strftime("%Y-%m-%d-%H")
        if now_utc.hour in DAILY_REPORT_HOURS_UTC and now_utc.minute == 0 and last_daily_report_slot != slot:
            threading.Thread(target=job_daily_report, args=(pipeline,), daemon=True).start()
            last_daily_report_slot = slot

        if pipeline.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
            pipeline.logger.error("Too many consecutive errors. Shutting down.")
            break

        time.sleep(10)  # check every 10 seconds

    pipeline.logger.info("Sleep-loop scheduler stopped")


# ── Signal handling ────────────────────────────────────────────────────────────

def _setup_signal_handlers():
    """Register SIGINT / SIGTERM handlers. Returns a threading.Event to watch."""
    import threading
    stop_event = threading.Event()

    def _handle(sig, frame):
        print(f"\nReceived signal {sig} — stopping scheduler...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    return stop_event


# ── Main entry point ───────────────────────────────────────────────────────────

def start_daemon() -> None:
    """
    Initialize pipeline, run the initial warm-up pass, then enter the
    recurring schedule loop.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("  Stock_ML Scheduler — starting up")
    logger.info(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 60)

    pipeline = SchedulerPipeline()

    # Run all jobs once immediately on startup
    logger.info("Running initial warm-up pass...")
    _run_all_jobs(pipeline)
    logger.info("Warm-up complete — entering recurring schedule")

    if _HAS_APSCHEDULER:
        run_with_apscheduler(pipeline)
    else:
        run_with_sleep_loop(pipeline)


if __name__ == "__main__":
    start_daemon()
