"""
Backtesting module for Stock_ML.

Provides:
    BacktestEngine       — runs simulated portfolio over historical data
    PortfolioSimulator   — tracks cash, positions, NAV
    PerformanceCalculator — computes comprehensive performance statistics
    BacktestReporter      — outputs results to console, CSV, and HTML
"""

from backtest.engine import BacktestEngine, BacktestResult
from backtest.portfolio import PortfolioSimulator
from backtest.performance import PerformanceCalculator, PerformanceReport
from backtest.reporter import BacktestReporter

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "PortfolioSimulator",
    "PerformanceCalculator",
    "PerformanceReport",
    "BacktestReporter",
]
