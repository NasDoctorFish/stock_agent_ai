"""Plunge detector (optional feature, from the Plunge Detector project).

A symbol alerts when its latest price is >= drop_pct_threshold below its
N-day high (prior closes only), outside its per-symbol cooldown, and (if
market_hours_only) while its exchange is open. Alerts go to Telegram and
Discord and are kept in data/plunge/alerts.json; per-symbol last-check state
in data/plunge/status.json (read by the web API).

Enable in the daemon with PLUNGE_ENABLED=1 (poll interval from watchlist.yaml).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

import pandas as pd

from plunge.config import WatchlistConfig, load_watchlist_config

logger = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parents[2] / "data" / "plunge"
ALERTS, STATUS = DATA / "alerts.json", DATA / "status.json"
MAX_ALERTS = 500


def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write(path: Path, obj) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def market_open(symbol: str, now: datetime) -> bool:
    kr = symbol.endswith((".KS", ".KQ"))
    local = now.astimezone(ZoneInfo("Asia/Seoul" if kr else "America/New_York"))
    start, end = ((9, 0), (15, 30)) if kr else ((9, 30), (16, 0))
    return local.weekday() < 5 and start <= (local.hour, local.minute) <= end


def evaluate(symbol: str, closes: pd.Series, lookback_days: int, threshold: float):
    """closes: ascending daily closes, last = latest price. Returns alert dict or None."""
    closes = closes.dropna()
    if len(closes) < 2:
        return None
    cur, ref = float(closes.iloc[-1]), float(closes.iloc[:-1].tail(lookback_days).max())
    drop = (ref - cur) / ref * 100
    if drop < threshold:
        return None
    return {"symbol": symbol, "current_price": cur, "reference_price": ref,
            "reference_type": "n_day_high", "drop_pct": round(drop, 2), "threshold_pct": threshold}


def _notify(a: dict) -> None:
    text = (f"PLUNGE {a['symbol']}: -{a['drop_pct']}% from {a['lookback']}d high "
            f"({a['reference_price']:.2f} -> {a['current_price']:.2f})")
    try:
        from reporters.telegram_sender import TelegramSender
        TelegramSender().send(text)
    except Exception as exc:
        logger.warning(f"plunge telegram failed: {exc}")
    try:
        from reporters.discord_notifier import DiscordWebhookNotifier
        DiscordWebhookNotifier().send_text(text)
    except Exception as exc:
        logger.warning(f"plunge discord failed: {exc}")


def scan(cfg: WatchlistConfig | None = None, notify: bool = True, now: datetime | None = None) -> List[dict]:
    """One poll over the watchlist. Returns new alerts."""
    import yfinance as yf

    cfg = cfg or load_watchlist_config()
    now = now or datetime.now(timezone.utc)
    syms = [t.symbol for t in cfg.tickers]
    days = max(t.lookback_days for t in cfg.tickers) + 10
    px = yf.download(syms, period=f"{days * 2}d", interval="1d", auto_adjust=True,
                     progress=False, group_by="column")["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(syms[0])

    history: List[dict] = _read(ALERTS, [])
    status: Dict[str, dict] = _read(STATUS, {})
    new: List[dict] = []
    for t in cfg.tickers:
        if t.symbol not in px:
            continue
        closes = px[t.symbol].dropna()
        if closes.empty:
            continue
        status[t.symbol] = {"symbol": t.symbol, "current_price": float(closes.iloc[-1]),
                            "last_checked": now.isoformat()}
        if t.market_hours_only and not market_open(t.symbol, now):
            continue
        a = evaluate(t.symbol, closes, t.lookback_days, t.drop_pct_threshold)
        if not a:
            continue
        last = next((h for h in reversed(history) if h["symbol"] == t.symbol), None)
        if last and (now - datetime.fromisoformat(last["triggered_at"])).total_seconds() < t.cooldown_minutes * 60:
            continue
        a.update(triggered_at=now.isoformat(), lookback=t.lookback_days)
        history.append(a)
        new.append(a)
        if notify:
            _notify(a)
    _write(ALERTS, history[-MAX_ALERTS:])
    _write(STATUS, status)
    return new


def get_alerts(limit: int = 50) -> List[dict]:
    return _read(ALERTS, [])[-limit:][::-1]


def get_status() -> Dict[str, dict]:
    return _read(STATUS, {})
