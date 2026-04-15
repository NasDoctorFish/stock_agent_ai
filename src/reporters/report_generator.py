"""
3-Hour Report Generator
========================
Assembles a structured status report from all pipeline components and
delivers it via Discord webhook (with local file fallback).

Report sections (each becomes a Discord embed):
  1. Header summary  — timestamp, cycle, paper/live mode
  2. US Market Regime
  3. KR Market Regime
  4. Market Valuation
  5. US Top Stocks
  6. KR Top Stocks
  7. Orders This Cycle

The full report is also saved as a Markdown file under reports/.

Notification channel:
  Uses DiscordWebhookNotifier (primary: DISCORD_WEBHOOK_Claude_AI,
  fallback: DISCORD_WEBHOOK).  No bot token is used.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from reporters.discord_notifier import (
    DiscordWebhookNotifier,
    COLOR_BLUE, COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_GREY,
    regime_embed,
    valuation_embed,
    decision_embed,
    orders_embed,
)

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


class ReportGenerator:
    """
    Builds and delivers the 3-hour status report via Discord webhook.

    Args:
        notifier: DiscordWebhookNotifier instance (auto-created from env if None)
    """

    def __init__(self, notifier: Optional[DiscordWebhookNotifier] = None):
        self._notifier = notifier or DiscordWebhookNotifier()
        self._cycle = 0

    # ── Public entry point ─────────────────────────────────────────────────────

    def generate_and_send(
        self,
        us_regime: Optional[Dict] = None,
        kr_regime: Optional[Dict] = None,
        market_timing: Optional[Dict] = None,
        us_decision: Optional[Dict] = None,
        kr_decision: Optional[Dict] = None,
        order_log: Optional[List[Dict]] = None,
        paper_trading: bool = True,
    ) -> str:
        """
        Build the full report, save it as Markdown, and send to Discord.

        Args:
            us_regime:     Output of USRegimeDetector.detect()
            kr_regime:     Output of KRRegimeDetector.detect()
            market_timing: Output of MarketTimer.run()
            us_decision:   Output of DecisionMaker.run() for US pipeline
            kr_decision:   Output of KRDecisionMaker.run() for KR pipeline
            order_log:     Combined list of order dicts from all brokers
            paper_trading: Whether current run is in paper mode (shown in header)

        Returns:
            The full report as a Markdown string.
        """
        self._cycle += 1
        now_utc = datetime.utcnow()
        now_kst = now_utc + timedelta(hours=9)
        ts_utc = now_utc.strftime("%Y-%m-%d %H:%M UTC")
        ts_kst = now_kst.strftime("%Y-%m-%d %H:%M KST")
        mode = "📄 PAPER" if paper_trading else "🔴 LIVE"

        # ── Discord embeds ────────────────────────────────────────────────────
        header = {
            "title": f"📊 Stock_ML Report — Cycle {self._cycle}",
            "description": (
                f"**{ts_utc}**  ({ts_kst})\n"
                f"Mode: {mode}"
            ),
            "color": COLOR_BLUE,
            "timestamp": now_utc.isoformat(),
        }

        embeds = [
            header,
            regime_embed(us_regime, "US"),
            regime_embed(kr_regime, "KR"),
            valuation_embed(market_timing),
            decision_embed(us_decision, "US"),
            decision_embed(kr_decision, "KR"),
            orders_embed(order_log),
        ]

        self._notifier.send_embeds(embeds)

        # ── Markdown file ─────────────────────────────────────────────────────
        report_md = self._build_markdown(
            ts_utc, ts_kst, mode,
            us_regime, kr_regime, market_timing,
            us_decision, kr_decision, order_log,
        )
        self._save_report(report_md, ts_utc)
        return report_md

    # ── Markdown builder (for file persistence) ───────────────────────────────

    def _build_markdown(
        self,
        ts_utc: str,
        ts_kst: str,
        mode: str,
        us_regime, kr_regime, market_timing,
        us_decision, kr_decision, order_log,
    ) -> str:
        lines: List[str] = [
            f"# Stock_ML Report — Cycle {self._cycle}",
            f"**{ts_utc}** ({ts_kst}) | Mode: {mode}",
            "",
            "## 🇺🇸 US Market Regime",
        ]
        lines += _md_regime(us_regime)
        lines += ["", "## 🇰🇷 KR Market Regime"]
        lines += _md_regime(kr_regime)
        lines += ["", "## 💰 Market Valuation"]
        lines += _md_valuation(market_timing)
        lines += ["", "## 📈 US Top Stocks"]
        lines += _md_decision(us_decision)
        lines += ["", "## 🏢 KR Top Stocks"]
        lines += _md_decision(kr_decision)
        lines += ["", "## 📋 Orders This Cycle"]
        lines += _md_orders(order_log)
        lines += ["", "---", "_Stock_ML Automated Report_"]
        return "\n".join(lines)

    def _save_report(self, text: str, ts_str: str) -> None:
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            safe_ts = ts_str.replace(":", "").replace(" ", "_")
            path = REPORTS_DIR / f"report_{safe_ts}.md"
            path.write_text(text, encoding="utf-8")
            logger.info(f"[Report] Saved: {path}")
        except Exception as exc:
            logger.error(f"[Report] Could not save file: {exc}")


# ── Markdown formatting helpers ────────────────────────────────────────────────

def _md_regime(regime: Optional[Dict]) -> List[str]:
    if not regime:
        return ["*No data*"]
    score = regime.get("score", 0)
    label = regime.get("regime", "unknown")
    as_of = regime.get("as_of", "?")
    signals = regime.get("signals", {})
    err = regime.get("error")
    lines = [
        f"**Regime:** {label}",
        f"**Score:** {score:+d}/6",
        f"**As of:** {as_of}",
    ]
    if signals:
        lines.append("**Signals:**")
        for k, v in signals.items():
            icon = "✅" if v == 1 else "❌"
            lines.append(f"  {icon} {k.replace('_', ' ')}")
    if err:
        lines.append(f"⚠️ {err[:200]}")
    return lines


def _md_valuation(timing: Optional[Dict]) -> List[str]:
    if not timing:
        return ["*No data*"]
    lines = []
    for mkt in ["us", "kr"]:
        d = timing.get(mkt, {})
        if not d:
            continue
        label = d.get("label", "N/A")
        pe = d.get("pe_ratio")
        spread = d.get("fed_model_spread_pp")
        lines.append(f"**{mkt.upper()}:** {label}" + (f"  (P/E {pe:.1f}x)" if pe else ""))
        if spread is not None:
            lines.append(f"  Earnings yield spread vs bonds: {spread:+.1f}pp")
    return lines or ["*No data*"]


def _md_decision(decision: Optional[Dict]) -> List[str]:
    if not decision:
        return ["*No data*"]
    regime = decision.get("economic_regime", "unknown")
    top = decision.get("top_stocks", [])[:5]
    rec = decision.get("recommendation", "")
    lines = [f"**Regime:** `{regime}`", f"_{rec[:120]}_"]
    if top:
        lines.append("**Top picks:**")
        for s in top:
            sid = s.get("stock_id", "?")
            kcode = s.get("kiwoom_code", "")
            score = s.get("score", 0.0)
            reasons = ", ".join(s.get("reasons", [])[:2]) or "—"
            code_str = f" ({kcode})" if kcode else ""
            lines.append(f"  - `{sid}`{code_str}  score={score:+.3f}  {reasons}")
    return lines


def _md_orders(order_log: Optional[List[Dict]]) -> List[str]:
    if not order_log:
        return ["*No orders placed*"]
    lines = []
    for o in order_log[:10]:
        side = o.get("side", "?").upper()
        ticker = o.get("ticker") or o.get("stock_code") or o.get("currency", "?")
        qty = o.get("quantity", "?")
        price = o.get("price")
        status = o.get("status", "?")
        paper = " [PAPER]" if o.get("paper_trading") else ""
        price_str = f"@ {price:,.0f}" if price else "(market)"
        lines.append(f"  - {side} `{ticker}` ×{qty} {price_str} → `{status}`{paper}")
    return lines


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)

    gen = ReportGenerator()
    sample_us_regime = {
        "market": "US", "regime": "Cautious (Late Cycle / Mixed)",
        "score": 2,
        "signals": {
            "sp_trend": 1, "yield_curve": -1, "vix": 1,
            "credit_spread": 1, "liquidity": -1, "momentum": 1,
        },
        "as_of": "2026-04-06", "error": None,
    }
    report = gen.generate_and_send(us_regime=sample_us_regime, paper_trading=True)
    print("\n--- REPORT PREVIEW (first 500 chars) ---")
    print(report[:500])
