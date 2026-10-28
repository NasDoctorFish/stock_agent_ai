"""
Watchlist Config
================
Loads ticker watchlist from config/watchlist.yaml into typed dataclasses.

Moved from Feb_08_2026 (Plunge Detector); default path is src/config/watchlist.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "watchlist.yaml"


@dataclass
class TickerConfig:
    symbol: str
    lookback_days: int = 30
    drop_pct_threshold: float = 5.0
    cooldown_minutes: int = 60
    market_hours_only: bool = True


@dataclass
class WatchlistConfig:
    poll_interval_minutes: int = 5
    plunge_summary_hours: float = 24.0
    tickers: List[TickerConfig] = field(default_factory=list)


def load_watchlist_config(path: Optional[Path] = None) -> WatchlistConfig:
    path = path or DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text("utf-8"))
    tickers = [
        TickerConfig(
            symbol=entry["symbol"],
            lookback_days=entry.get("lookback_days", 30),
            drop_pct_threshold=entry.get("drop_pct_threshold", 5.0),
            cooldown_minutes=entry.get("cooldown_minutes", 60),
            market_hours_only=entry.get("market_hours_only", True),
        )
        for entry in raw.get("tickers", [])
    ]
    return WatchlistConfig(
        poll_interval_minutes=raw.get("poll_interval_minutes", 5),
        plunge_summary_hours=raw.get("plunge_summary_hours", 24.0),
        tickers=tickers,
    )
