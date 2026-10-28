"""
run_backtest.py — CLI entry point for the Stock_ML backtesting module.

Usage
-----
python src/backtest/run_backtest.py \
    --start 2022-01-01 \
    --end   2025-01-01 \
    --rebalance weekly \
    --top-n 10 \
    --initial-capital 100000000 \
    --benchmark SPY \
    --output reports/

Optional flags:
    --no-html    Skip HTML report generation
    --no-csv     Skip CSV report generation
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

# ── Ensure src/ is on sys.path so all imports resolve correctly ───────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.engine import BacktestEngine
from backtest.performance import PerformanceCalculator
from backtest.reporter import BacktestReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock_ML Backtester — walks forward through historical data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start",            default="2022-01-01",   help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end",              default="2025-01-01",   help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--rebalance",        default="monthly",      choices=["weekly", "biweekly", "monthly", "quarterly", "custom"],
                        help="Rebalancing frequency")
    parser.add_argument("--rebalance-days",   default=21, type=int,   help="Period in days when --rebalance custom")
    parser.add_argument("--top-n",            default=10,  type=int,  help="Number of stocks to hold")
    parser.add_argument("--initial-capital",  default=100_000_000, type=float,
                        help="Starting portfolio value")
    parser.add_argument("--benchmark",        default="SPY",          help="Benchmark ticker in DuckDB")
    parser.add_argument("--db",               default=None,           help="DuckDB path (default src/db/stockml.duckdb; use data/research/research.duckdb for long history)")
    parser.add_argument("--output",           default="reports/",     help="Output directory for reports")
    parser.add_argument("--no-html",          action="store_true",    help="Skip HTML report")
    parser.add_argument("--no-csv",           action="store_true",    help="Skip CSV report")
    parser.add_argument("--log-level",        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    if args.rebalance == "custom":
        args.rebalance = f"{args.rebalance_days}d"
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # ── Locate DuckDB ─────────────────────────────────────────────────────────
    # Resolve relative to the project root (two levels above src/)
    project_root = Path(__file__).resolve().parents[2]
    db_path = Path(args.db) if args.db else project_root / "src" / "db" / "stockml.duckdb"

    if not db_path.exists():
        logger.error(f"DuckDB not found at {db_path}. Ensure the database is initialised first.")
        sys.exit(1)

    logger.info(f"Using database: {db_path}")
    logger.info(
        f"Config: {args.start} → {args.end} | freq={args.rebalance} | "
        f"top_n={args.top_n} | capital={args.initial_capital:,.0f} | bench={args.benchmark}"
    )

    # ── Run backtest ──────────────────────────────────────────────────────────
    engine = BacktestEngine(
        db_path=str(db_path),
        start_date=args.start,
        end_date=args.end,
        rebalance_freq=args.rebalance,
        top_n=args.top_n,
        initial_capital=args.initial_capital,
        benchmark_ticker=args.benchmark,
    )

    try:
        result = engine.run()
    except Exception as exc:
        logger.error(f"Backtest failed: {exc}", exc_info=True)
        sys.exit(1)

    # ── Check we have sufficient data ────────────────────────────────────────
    if result.nav_series.empty:
        logger.warning("NAV series is empty — no tradeable data found in the given date range.")
        print("Warning: No data available for the specified date range.")
        print("Backtest complete (no results). Check your database and date range.")
        return

    if len(result.nav_series) < 5:
        logger.warning(f"Very few NAV data points ({len(result.nav_series)}). Results may be unreliable.")

    # ── Compute performance ───────────────────────────────────────────────────
    logger.info("Computing performance statistics...")
    calc = PerformanceCalculator(
        nav=result.nav_series,
        benchmark=result.benchmark_series,
        trade_log=result.trade_log,
        regime_history=result.regime_history,
    )
    try:
        report = calc.compute()
    except Exception as exc:
        logger.warning(f"Performance calculation encountered an error: {exc}", exc_info=True)
        from backtest.performance import PerformanceReport
        report = PerformanceReport()

    # ── Console output ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BACKTEST PERFORMANCE REPORT")
    print("=" * 60)
    BacktestReporter.to_console(report)
    print()

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir = args.output
    saved_paths = []

    if not args.no_csv:
        try:
            csv_path = BacktestReporter.to_csv(result, report, output_dir)
            saved_paths.append(csv_path)
            logger.info(f"CSV saved: {csv_path}")
        except Exception as exc:
            logger.warning(f"CSV export failed: {exc}")

    if not args.no_html:
        try:
            html_path = BacktestReporter.to_html(result, report, output_dir)
            if html_path:
                saved_paths.append(html_path)
                logger.info(f"HTML saved: {html_path}")
            else:
                logger.info("HTML skipped (plotly not installed).")
        except Exception as exc:
            logger.warning(f"HTML export failed: {exc}")

    print(f"Backtest complete. Reports saved to {output_dir}")
    if saved_paths:
        for p in saved_paths:
            print(f"  → {p}")


if __name__ == "__main__":
    main()
