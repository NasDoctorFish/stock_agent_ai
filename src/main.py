"""
Stock_ML — Entry Point
======================
Modes of operation:

  python main.py              # Run once: US + KR analysis + print decisions
  python main.py --daemon     # Start 24/7 scheduler
  python main.py --kr-only    # Run KR pipeline only (skip US)
  python main.py --us-only    # Run US pipeline only (skip KR)
  python main.py --web        # Start Vue dashboard web server (port 8080)
  python main.py --web --port 9090  # Web server on custom port
  python main.py --paper      # Explicitly force paper trading (default)
  python main.py --live       # Enable LIVE trading — USE WITH CAUTION

Safety default:
  paper_trading=True is the hard default.  Pass --live only when you are
  ready to send real orders through Kiwoom.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is on sys.path when invoked as a script
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import os
from dotenv import load_dotenv

load_dotenv()

from colorama import Fore, init as colorama_init
colorama_init(autoreset=True)

from core.db_handler import DBHandler
from core.db_updater import DBUpdater
from core.status_logger import StatusLogger
from core.decision_maker import DecisionMaker
from core.kr_decision_maker import KRDecisionMaker
from core.investor import Investor

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
from reporters.discord_notifier import DiscordWebhookNotifier

from brokers.kiwoom_broker import KiwoomBroker
from brokers.kr_order_executor import KROrderExecutor

logger_sys = logging.getLogger(__name__)


# ── US Pipeline ───────────────────────────────────────────────────────────────

def run_us_pipeline(db, logger) -> dict:
    """Run the full US analysis pipeline and return the decision dict."""
    logger.info("=== US Pipeline ===")

    macro = MacroAnalyst(db, logger)
    macro_result = macro.run()

    # Optional: visualise macro trend for BRENT
    df_feat = macro_result.artifacts.get("df_feat")
    if df_feat is not None:
        try:
            macro.visualize_macro_trend(df_feat, "BRENT")
        except Exception:
            pass

    fund = FundamentalAnalyst(db, logger)
    stock = StockDataAnalyst(db, logger)
    news = NewsMediaAnalyst(db, logger)
    biz = BusinessAnalyst(db, logger)

    investor = Investor([macro, fund, stock, news, biz], logger)
    # US uses the original DecisionMaker (reads "stock_data" key)
    decision_maker = DecisionMaker(investor, logger)
    decision = decision_maker.run()

    print(Fore.CYAN + "\n[US] Final Investment Decision")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return decision


# ── KR Pipeline ───────────────────────────────────────────────────────────────

def run_kr_pipeline(logger, paper_trading: bool = True) -> tuple[dict, list]:
    """
    Run the Korean stock analysis pipeline.

    Steps:
      1. KRStockDataAnalyst — fetch live yfinance data, compute technicals
      2. KRDecisionMaker    — score stocks, detect regime, build decision
      3. KROrderExecutor    — generate + execute orders via KiwoomBroker

    Args:
        logger:        StatusLogger instance
        paper_trading: True = log orders only (safe default)
                       False = send real orders through Kiwoom API

    Returns:
        (decision dict, list of order dicts placed this cycle)
    """
    logger.info("=== KR Pipeline ===")

    # ── 1. Analysis ───────────────────────────────────────────────────────────
    kr_analyst = KRStockDataAnalyst(logger_=logger)
    investor = Investor([kr_analyst], logger)

    # KRDecisionMaker reads "kr_stock_data" — the actual key KRStockDataAnalyst produces
    kr_dm = KRDecisionMaker(investor, logger_=logger)
    decision = kr_dm.run()

    print(Fore.YELLOW + "\n[KR] Final Investment Decision")
    print(json.dumps(decision, indent=2, ensure_ascii=False, default=str))

    # ── 2. Order execution ────────────────────────────────────────────────────
    order_log: list = []
    try:
        broker = KiwoomBroker(paper_trading=paper_trading)
        executor = KROrderExecutor(
            broker=broker,
            max_signals=3,          # max 3 buy orders per cycle
            min_score_threshold=0.0,  # accept any positive score
            default_quantity=1,     # 1 share per signal (safe for paper/live)
        )
        order_log = executor.execute(decision)
        logger.info(f"[KR] {len(order_log)} order(s) executed (paper={paper_trading})")
    except ValueError as exc:
        # KiwoomBroker raises ValueError if credentials are missing
        logger.warn(f"[KR] Kiwoom credentials not set — skipping order execution: {exc}")
    except Exception as exc:
        logger.error(f"[KR] Order execution error: {exc}", exc_info=True)

    return decision, order_log


# ── Regime + Timing ───────────────────────────────────────────────────────────

def run_regime_and_timing(logger) -> tuple[dict, dict, dict]:
    """Run US regime, KR regime, and market timing. Return all three."""
    fred_key = os.getenv("FRED_API_KEY")

    logger.info("Detecting US market regime...")
    us_regime = USRegimeDetector(fred_api_key=fred_key).detect()
    logger.info(f"  US regime: {us_regime['regime']} (score {us_regime['score']:+d})")

    logger.info("Detecting KR market regime...")
    kr_regime = KRRegimeDetector(fred_api_key=fred_key).detect()
    logger.info(f"  KR regime: {kr_regime['regime']} (score {kr_regime['score']:+d})")

    logger.info("Running market timing / valuation...")
    timing = MarketTimer(fred_api_key=fred_key).run()
    logger.info(
        f"  US valuation: {timing['us']['label']} | "
        f"KR valuation: {timing['kr']['label']}"
    )

    return us_regime, kr_regime, timing


# ── One-shot run ──────────────────────────────────────────────────────────────

def run_once(
    us_only: bool = False,
    kr_only: bool = False,
    paper_trading: bool = True,
) -> None:
    """
    Run the full pipeline once and exit.

    Args:
        us_only:       Skip KR pipeline
        kr_only:       Skip US pipeline AND data update (KR fetches live from yfinance)
        paper_trading: True (default) = paper mode; False = live orders
    """
    logger = StatusLogger("System")

    us_decision = None
    kr_decision = None
    order_log: list = []

    if not kr_only:
        # Data update is only needed for the US pipeline (DuckDB / CSV)
        db = DBHandler(logger)
        updater = DBUpdater(db, logger)
        logger.info("Updating data...")
        updater.update_csv_files()
        db.save_all_to_duckdb()
    else:
        db = None

    # Regime + timing always run (fast, independent of DB)
    us_regime, kr_regime, timing = run_regime_and_timing(logger)

    if not kr_only and db is not None:
        us_decision = run_us_pipeline(db, logger)

    if not us_only:
        kr_decision, order_log = run_kr_pipeline(logger, paper_trading=paper_trading)

    # Send report via Discord webhook
    reporter = ReportGenerator(notifier=DiscordWebhookNotifier())
    reporter.generate_and_send(
        us_regime=us_regime,
        kr_regime=kr_regime,
        market_timing=timing,
        us_decision=us_decision,
        kr_decision=kr_decision,
        order_log=order_log,
        paper_trading=paper_trading,
    )


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock_ML — Automated Investment Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--daemon", action="store_true",
                      help="Start 24/7 scheduler")
    mode.add_argument("--us-only", action="store_true",
                      help="Run US pipeline only")
    mode.add_argument("--kr-only", action="store_true",
                      help="Run KR pipeline only (no DB update, fetches live from yfinance)")
    mode.add_argument("--web", action="store_true",
                      help="Start Vue dashboard web server")

    parser.add_argument("--port", type=int, default=8080,
                        help="Web server port (used with --web, default 8080)")

    trading = parser.add_mutually_exclusive_group()
    trading.add_argument("--paper", action="store_true", default=True,
                         help="Paper trading mode — orders logged but not sent (default)")
    trading.add_argument("--live", action="store_true",
                         help="LIVE trading — real orders sent to Kiwoom (use with care!)")

    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()
    paper_trading = not args.live  # live flag overrides paper

    if paper_trading:
        print(Fore.GREEN + "[main] Running in PAPER TRADING mode — no real orders will be sent")
    else:
        print(Fore.RED + "[main] *** LIVE TRADING MODE — real orders WILL be sent via Kiwoom ***")

    if args.daemon:
        from scheduler.scheduler import start_daemon
        start_daemon()
    elif args.web:
        from web.server import make_app
        import tornado.ioloop
        app = make_app()
        app.listen(args.port)
        print(Fore.CYAN + f"[main] Dashboard running at http://localhost:{args.port}")
        tornado.ioloop.IOLoop.current().start()
    elif args.us_only:
        run_once(us_only=True, paper_trading=paper_trading)
    elif args.kr_only:
        run_once(kr_only=True, paper_trading=paper_trading)
    else:
        run_once(paper_trading=paper_trading)


if __name__ == "__main__":
    main()
