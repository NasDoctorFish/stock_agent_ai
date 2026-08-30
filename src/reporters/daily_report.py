"""
Daily Market Report
====================
Separate from the 3-hour ML pipeline report (report_generator.py). Gathers
market snapshot / watchlist / AI-signal / news / risk / calendar data,
hands it to Claude as a single formatting call, and sends the resulting
plain-text message via Telegram.

Required environment variables (.env), in addition to what report_generator.py
already needs:
    ANTHROPIC_API_KEY   — for the formatting call (client must set this)
    NEWS_API_KEY         — already used elsewhere (Market Drivers headlines)
    FMP_API_KEY           — already stored, used here for Tomorrow's calendar

If ANTHROPIC_API_KEY is missing or the Claude call fails for any reason,
a plain-Python fallback formatter is used instead so a report always ships.

Standalone test:
    poetry run python3 -m src.reporters.daily_report
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_FEB_SRC = Path(__file__).resolve().parents[2] / "Feb_08_2026" / "src"
if str(_FEB_SRC) not in sys.path:
    sys.path.append(str(_FEB_SRC))  # after _SRC — root's utils/* must win on name clash

import requests
import yfinance as yf
import anthropic
from dotenv import load_dotenv

load_dotenv()

from reporters.telegram_sender import TelegramSender
from regime_detector.us_regime import USRegimeDetector
from utils.fetch_newsapi import fetch_news_text_only
from monitors.watchlist_config import load_watchlist_config

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
WATCHLIST_LIMIT = 8
MODEL_ID = "claude-opus-5"

_SNAPSHOT_TICKERS = {
    "sp500": "^GSPC",
    "nasdaq100": "^NDX",
    "kospi": "^KS11",
    "gold": "GC=F",
    "us10y": "^TNX",
}


# ── data gathering ───────────────────────────────────────────────────────────

def _load_watchlist_tickers(limit: int = WATCHLIST_LIMIT) -> List[str]:
    try:
        cfg = load_watchlist_config()
        tickers = [t.symbol for t in cfg.tickers if not t.symbol.endswith(".KS")]
        return tickers[:limit]
    except Exception as exc:
        logger.error(f"[DailyReport] watchlist load failed: {exc}")
        return []


def gather_market_snapshot(us_regime: Optional[Dict] = None) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    try:
        data = yf.download(
            list(_SNAPSHOT_TICKERS.values()), period="5d", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
        )
        for name, ticker in _SNAPSHOT_TICKERS.items():
            try:
                closes = data[ticker]["Close"].dropna()
                if len(closes) < 2:
                    continue
                last, prev = closes.iloc[-1], closes.iloc[-2]
                if name == "us10y":
                    # ^TNX close is already the actual yield in percent (e.g. 4.70 -> 4.70%)
                    snapshot["us10y"] = {
                        "yield_pct": round(float(last), 2),
                        "bp_change": round((last - prev) * 100, 1),
                    }
                else:
                    snapshot[name] = {
                        "price": round(float(last), 2),
                        "pct_change": round((last / prev - 1) * 100, 2),
                    }
            except Exception as exc:
                logger.warning(f"[DailyReport] snapshot ticker {ticker} failed: {exc}")
    except Exception as exc:
        logger.error(f"[DailyReport] market snapshot fetch failed: {exc}")

    if us_regime is None:
        try:
            us_regime = USRegimeDetector(fred_api_key=os.getenv("FRED_API_KEY")).detect()
        except Exception as exc:
            logger.error(f"[DailyReport] regime detect for mood failed: {exc}")
            us_regime = None

    if us_regime:
        score = us_regime.get("score", 0)
        snapshot["market_mood"] = {
            "label": us_regime.get("regime", "unknown"),
            "emoji": "🟢" if score >= 4 else "🟡" if score >= 0 else "🔴",
        }
    return snapshot


def classify_signal(
    momentum_5d_pct: float,
    above_ma20: bool,
    above_ma60: bool,
    decision_score: Optional[float] = None,
) -> Tuple[str, int]:
    """Pure heuristic classifier — NOT a calibrated ML probability."""
    raw = momentum_5d_pct * 4
    raw += 5 if above_ma20 else -5
    raw += 5 if above_ma60 else -5
    if decision_score is not None:
        raw += decision_score * 10

    confidence_pct = int(max(0, min(100, round(50 + raw))))
    if raw > 10:
        label = "BULLISH"
    elif raw < -10:
        label = "BEARISH"
    else:
        label = "NEUTRAL"
    return label, confidence_pct


def gather_watchlist_and_signals(
    tickers: List[str],
    decision_top_stocks: Optional[List[Dict]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    watchlist: List[Dict] = []
    ai_signals: List[Dict] = []
    if not tickers:
        return watchlist, ai_signals

    score_by_ticker = {}
    reasons_by_ticker = {}
    for s in decision_top_stocks or []:
        sid = s.get("stock_id")
        if sid:
            score_by_ticker[sid] = s.get("score")
            reasons_by_ticker[sid] = s.get("reasons", [])[:2]

    try:
        data = yf.download(
            tickers, period="4mo", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
        )
    except Exception as exc:
        logger.error(f"[DailyReport] watchlist fetch failed: {exc}")
        return watchlist, ai_signals

    for t in tickers:
        try:
            closes = data[t]["Close"].dropna() if len(tickers) > 1 else data["Close"].dropna()
            if len(closes) < 2:
                continue
            last, prev = closes.iloc[-1], closes.iloc[-2]
            watchlist.append({
                "ticker": t,
                "price": round(float(last), 2),
                "pct_change": round((last / prev - 1) * 100, 2),
            })

            momentum_5d = (last / closes.iloc[-6] - 1) * 100 if len(closes) >= 6 else 0.0
            ma20 = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else None
            ma60 = closes.rolling(60).mean().iloc[-1] if len(closes) >= 60 else None
            above_ma20 = bool(ma20 is not None and last > ma20)
            above_ma60 = bool(ma60 is not None and last > ma60)

            label, confidence = classify_signal(
                momentum_5d, above_ma20, above_ma60, score_by_ticker.get(t),
            )
            ai_signals.append({
                "ticker": t,
                "signal": label,
                "confidence_pct": confidence,
                "momentum_5d_pct": round(float(momentum_5d), 2),
                "trend_vs_ma20": "above" if ma20 is not None and above_ma20 else ("below" if ma20 is not None else "unavailable"),
                "trend_vs_ma60": "above" if ma60 is not None and above_ma60 else ("below" if ma60 is not None else "unavailable"),
                "decision_reasons": reasons_by_ticker.get(t, []),
            })
        except Exception as exc:
            logger.warning(f"[DailyReport] signal calc for {t} failed: {exc}")

    return watchlist, ai_signals


def gather_market_drivers(tickers: List[str]) -> List[Dict]:
    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
        return []
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        df = fetch_news_text_only(list(tickers[:6]) + ["SPY", "QQQ"], yesterday, today, news_api_key)
        if df.empty:
            return []
        df = df.drop_duplicates(subset=["headline"]).head(15)
        return df[["headline", "source", "date"]].rename(columns={"headline": "title"}).to_dict("records")
    except Exception as exc:
        logger.error(f"[DailyReport] market drivers fetch failed: {exc}")
        return []


def gather_risk_watch(us_regime: Optional[Dict], us10y: Optional[Dict]) -> Dict:
    if not us_regime:
        return {}
    signals = us_regime.get("signals", {})
    risk = {
        "vix_signal": "low_fear" if signals.get("vix", 0) == 1 else "elevated_fear",
        "credit_spread_signal": "healthy" if signals.get("credit_spread", 0) == 1 else "stressed",
        "yield_curve_signal": "normal" if signals.get("yield_curve", 0) == 1 else "inverted",
        "overall_regime": us_regime.get("regime", "unknown"),
        "regime_score": us_regime.get("score", 0),
    }
    if us10y:
        risk["us10y_pct"] = us10y.get("yield_pct")
        risk["us10y_above_4pct"] = bool(us10y.get("yield_pct", 0) > 4.0)
    return risk


def gather_tomorrow(fmp_api_key: Optional[str]) -> Optional[Dict]:
    """FMP's calendar endpoints live under /stable/ now (the old /api/v3/
    economic_calendar and earning_calendar paths are retired "legacy" routes).
    Fetched independently — a plan restriction on one (e.g. economic-calendar
    needs a higher FMP tier) shouldn't drop the other."""
    if not fmp_api_key:
        return None
    tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    params = {"from": tomorrow, "to": tomorrow, "apikey": fmp_api_key}
    result: Dict[str, Any] = {}

    try:
        econ = requests.get(
            "https://financialmodelingprep.com/stable/economic-calendar",
            params=params, timeout=10,
        )
        econ.raise_for_status()
        data = econ.json()
        if isinstance(data, list):
            result["economic_events"] = data[:10]
        else:
            logger.warning(f"[DailyReport] economic-calendar unavailable: {data}")
    except Exception as exc:
        logger.warning(f"[DailyReport] economic-calendar fetch failed: {exc}")

    try:
        earnings = requests.get(
            "https://financialmodelingprep.com/stable/earnings-calendar",
            params=params, timeout=10,
        )
        earnings.raise_for_status()
        data = earnings.json()
        if isinstance(data, list):
            result["earnings"] = data[:10]
        else:
            logger.warning(f"[DailyReport] earnings-calendar unavailable: {data}")
    except Exception as exc:
        logger.warning(f"[DailyReport] earnings-calendar fetch failed: {exc}")

    return result or None


# ── Claude synthesis ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You format a daily market briefing for Telegram from structured JSON data.

Output ONLY the final plain-text message. No markdown code fences, no JSON, no commentary before or after.

Reproduce this exact structure (section order, emoji headers, and the divider line
"━━━━━━━━━━━━━━━━━━" between every section):

📊 DAILY MARKET REPORT
{date_str} | {time_str}
━━━━━━━━━━━━━━━━━━
🌎 MARKET SNAPSHOT
(one line per index in market_snapshot: name, price, ▲/▼ arrow, +/-% change;
 for us10y show the yield_pct and bp_change instead of price/%.
 Then a "Market Mood: {emoji} {label}" line.)
━━━━━━━━━━━━━━━━━━
🔥 WATCHLIST
(one line per watchlist ticker: ticker, price, ▲/▼ arrow, +/-% change.
 Then "⭐ Strongest:" and "⚠️ Weakest:" lines picking the max/min pct_change.)
━━━━━━━━━━━━━━━━━━
🤖 AI SIGNALS
(for each entry in ai_signals: ticker, Signal (🟢 BULLISH / 🟡 NEUTRAL / 🔴 BEARISH),
 Confidence (label it as a heuristic score, e.g. "Confidence (heuristic) NN%"),
 Momentum (momentum_5d_pct), Trend (trend_vs_ma20 / trend_vs_ma60).
 Do NOT include a 1-day forecast line — no forecasting model exists, never invent one.)
━━━━━━━━━━━━━━━━━━
📰 MARKET DRIVERS
(3-4 bullet points summarizing market_drivers_headlines. Write them STRICTLY from the
 supplied headlines — never invent a fact or event not present in the data.
 If market_drivers_headlines is empty, write exactly: "No major headlines available.")
━━━━━━━━━━━━━━━━━━
⚠️ RISK WATCH
(from risk_watch: one line each for vix_signal, credit_spread_signal, yield_curve_signal,
 and us10y_above_4pct if present. 🟢 for a healthy/normal/low_fear value, 🔴 otherwise.)
━━━━━━━━━━━━━━━━━━
🎯 TOMORROW
(list economic_events and earnings from the tomorrow field, if present.
 If tomorrow is null/empty, write exactly: "No scheduled events data available.")

Bot generated • {time_str}

Rules:
- Never invent, estimate, or round-guess any number not present in the input JSON.
- If a whole section's data is missing/empty, render it with a short "data unavailable"
  line instead of omitting the section header.
- Keep it compact — this is a Telegram message, not a report."""


def _call_claude(payload: Dict) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=2048,
        output_config={"effort": "medium"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Claude refused: {response.stop_details}")
    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text.strip():
        raise RuntimeError("Claude returned empty text")
    return text


def _fallback_plaintext(payload: Dict) -> str:
    lines = [
        "📊 DAILY MARKET REPORT",
        f"{payload['date_str']} | {payload['time_str']}",
        "━━━━━━━━━━━━━━━━━━",
        "🌎 MARKET SNAPSHOT",
    ]
    snap = payload.get("market_snapshot", {})
    for name, d in snap.items():
        if name == "market_mood":
            continue
        if name == "us10y":
            lines.append(f"US 10Y  {d.get('yield_pct')}%  ({d.get('bp_change'):+.1f}bp)")
        else:
            arrow = "▲" if d.get("pct_change", 0) >= 0 else "▼"
            lines.append(f"{name}  {d.get('price')}  {arrow} {d.get('pct_change'):+.2f}%")
    mood = snap.get("market_mood")
    if mood:
        lines.append(f"Market Mood: {mood['emoji']} {mood['label']}")

    lines += ["━━━━━━━━━━━━━━━━━━", "🔥 WATCHLIST"]
    for w in payload.get("watchlist", []):
        arrow = "▲" if w["pct_change"] >= 0 else "▼"
        lines.append(f"{w['ticker']}  ${w['price']}  {arrow} {w['pct_change']:+.2f}%")

    lines += ["━━━━━━━━━━━━━━━━━━", "🤖 AI SIGNALS"]
    for s in payload.get("ai_signals", []):
        lines.append(f"{s['ticker']}: {s['signal']} (confidence heuristic {s['confidence_pct']}%), "
                      f"momentum {s['momentum_5d_pct']:+.2f}%, MA20 {s['trend_vs_ma20']}, MA60 {s['trend_vs_ma60']}")

    lines += ["━━━━━━━━━━━━━━━━━━", "📰 MARKET DRIVERS"]
    headlines = payload.get("market_drivers_headlines", [])
    if headlines:
        for h in headlines[:4]:
            lines.append(f"• {h.get('title')}")
    else:
        lines.append("No major headlines available.")

    lines += ["━━━━━━━━━━━━━━━━━━", "⚠️ RISK WATCH"]
    risk = payload.get("risk_watch", {})
    for k, v in risk.items():
        lines.append(f"{k}: {v}")

    lines += ["━━━━━━━━━━━━━━━━━━", "🎯 TOMORROW"]
    tomorrow = payload.get("tomorrow")
    if tomorrow:
        lines.append(json.dumps(tomorrow, default=str)[:500])
    else:
        lines.append("No scheduled events data available.")

    lines += [f"\nBot generated • {payload['time_str']}"]
    return "\n".join(lines)


# ── Generator ─────────────────────────────────────────────────────────────────

class DailyReportGenerator:
    def __init__(self):
        self._telegram = TelegramSender(parse_mode=None)

    def build_payload(
        self,
        us_regime: Optional[Dict] = None,
        us_decision: Optional[Dict] = None,
        kr_decision: Optional[Dict] = None,
    ) -> Dict:
        now_utc = datetime.utcnow()
        now_sgt = now_utc + timedelta(hours=8)

        if us_regime is None:
            try:
                us_regime = USRegimeDetector(fred_api_key=os.getenv("FRED_API_KEY")).detect()
            except Exception as exc:
                logger.error(f"[DailyReport] regime detect failed: {exc}")
                us_regime = None

        market_snapshot = gather_market_snapshot(us_regime)
        tickers = _load_watchlist_tickers()
        top_stocks = (us_decision or {}).get("top_stocks", []) if us_decision else []
        watchlist, ai_signals = gather_watchlist_and_signals(tickers, top_stocks)
        market_drivers_headlines = gather_market_drivers(tickers)
        risk_watch = gather_risk_watch(us_regime, market_snapshot.get("us10y"))
        tomorrow = gather_tomorrow(os.getenv("FMP_API_KEY"))

        return {
            "date_str": now_sgt.strftime("%d %b %Y"),
            "time_str": now_sgt.strftime("%H:%M SGT"),
            "market_snapshot": market_snapshot,
            "watchlist": watchlist,
            "ai_signals": ai_signals,
            "market_drivers_headlines": market_drivers_headlines,
            "risk_watch": risk_watch,
            "tomorrow": tomorrow,
        }

    def generate_and_send(
        self,
        us_regime: Optional[Dict] = None,
        us_decision: Optional[Dict] = None,
        kr_decision: Optional[Dict] = None,
    ) -> str:
        payload = self.build_payload(us_regime, us_decision, kr_decision)
        try:
            text = _call_claude(payload)
        except Exception as exc:
            logger.error(f"[DailyReport] Claude synthesis failed: {exc} — using plaintext fallback")
            text = _fallback_plaintext(payload)

        self._save_report(text)
        self._telegram.send_chunked(text)
        return text

    def _save_report(self, text: str) -> None:
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = REPORTS_DIR / f"daily_report_{ts}.md"
            path.write_text(text, encoding="utf-8")
            logger.info(f"[DailyReport] Saved: {path}")
        except Exception as exc:
            logger.error(f"[DailyReport] Could not save file: {exc}")


# ── Self-test (offline, pure-function) ─────────────────────────────────────────

def _selftest() -> None:
    assert classify_signal(5.0, True, True)[0] == "BULLISH"
    assert classify_signal(-5.0, False, False)[0] == "BEARISH"
    assert classify_signal(0.1, True, False)[0] == "NEUTRAL"
    print("daily_report selftest OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
    gen = DailyReportGenerator()
    report = gen.generate_and_send()
    print("\n--- REPORT PREVIEW ---\n")
    print(report)
