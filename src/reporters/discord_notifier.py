"""
Discord Webhook Notifier
========================
Sends alerts and reports to Discord via webhook POST — no bot token required.

Why webhooks instead of bot token:
    The Discord bot (`Claude_AI_Application_Token`) is already connected to
    another live application.  Discord only allows one gateway connection per
    bot token; a second connection would drop the first.  Webhooks are
    completely independent of the bot connection — they are simple HTTP POSTs
    and can run from any number of processes simultaneously.

Environment variables (.env):
    DISCORD_WEBHOOK_Claude_AI   — Primary webhook (this application)
    DISCORD_WEBHOOK             — Fallback webhook

Both webhooks are tried in order; the fallback is used only if the primary
fails.  If both fail, the message is written to logs/discord_fallback.log.

Discord limits:
    - Webhook `content` field: 2 000 chars
    - Embed description:       4 096 chars
    - Embed field value:       1 024 chars
    - Total chars per message: 6 000 (across all embeds)
    - Embeds per message:      10

Long reports are split into multiple webhook calls automatically.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

FALLBACK_LOG = Path(__file__).resolve().parents[2] / "logs" / "discord_fallback.log"

# Discord embed colours (decimal)
COLOR_GREEN  = 0x2ECC71   # bullish / success
COLOR_RED    = 0xE74C3C   # bearish / error
COLOR_YELLOW = 0xF1C40F   # neutral / warning
COLOR_BLUE   = 0x3498DB   # informational
COLOR_GREY   = 0x95A5A6   # no data


class DiscordWebhookNotifier:
    """
    Sends messages to Discord via webhook.

    Args:
        webhook_url:     Primary webhook URL (defaults to DISCORD_WEBHOOK_Claude_AI)
        fallback_url:    Fallback webhook URL (defaults to DISCORD_WEBHOOK)
        username:        Display name shown in Discord (default "Stock_ML")
        avatar_url:      Optional avatar URL for the webhook message
        timeout:         HTTP request timeout in seconds (default 10)
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        fallback_url: Optional[str] = None,
        username: str = "Stock_ML",
        avatar_url: Optional[str] = None,
        timeout: int = 10,
    ):
        # Primary: app-specific webhook; Fallback: general webhook
        self._primary = webhook_url or os.getenv("DISCORD_WEBHOOK_Claude_AI", "").strip()
        self._fallback = fallback_url or os.getenv("DISCORD_WEBHOOK", "").strip()
        self._username = username
        self._avatar_url = avatar_url
        self._timeout = timeout

        if not self._primary and not self._fallback:
            logger.warning(
                "[Discord] Neither DISCORD_WEBHOOK_Claude_AI nor DISCORD_WEBHOOK "
                "is set — messages will only be written to fallback log"
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def send_text(self, content: str) -> bool:
        """
        Send a plain-text message (max 2 000 chars).
        Automatically chunks if content exceeds Discord's limit.
        """
        if len(content) <= 2000:
            return self._post({"content": content})
        # Chunk into 2000-char segments
        chunks = [content[i:i + 2000] for i in range(0, len(content), 2000)]
        return all(self._post({"content": chunk}) for chunk in chunks)

    def send_embed(
        self,
        title: str,
        description: str = "",
        color: int = COLOR_BLUE,
        fields: Optional[List[Dict[str, Any]]] = None,
        footer: str = "",
    ) -> bool:
        """
        Send a single Discord embed.

        Args:
            title:       Embed title (max 256 chars)
            description: Main body text (max 4 096 chars)
            color:       Sidebar colour (use COLOR_* constants)
            fields:      List of {"name": str, "value": str, "inline": bool}
            footer:      Footer text
        """
        embed: Dict[str, Any] = {
            "title": title[:256],
            "description": description[:4096],
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if fields:
            # Truncate each field value to 1 024 chars
            embed["fields"] = [
                {
                    "name": f.get("name", "")[:256],
                    "value": str(f.get("value", ""))[:1024],
                    "inline": f.get("inline", False),
                }
                for f in fields[:25]  # max 25 fields per embed
            ]
        if footer:
            embed["footer"] = {"text": footer[:2048]}

        return self._post({"embeds": [embed]})

    def send_embeds(self, embeds: List[Dict[str, Any]]) -> bool:
        """
        Send multiple embeds in batches of 10 (Discord's per-message limit).
        Each embed must already be a valid Discord embed dict.
        """
        if not embeds:
            return True
        success = True
        for i in range(0, len(embeds), 10):
            batch = embeds[i:i + 10]
            success = self._post({"embeds": batch}) and success
        return success

    def send_report(self, embeds: List[Dict[str, Any]], summary: str = "") -> bool:
        """
        Send a structured report: optional summary text + embed list.

        Args:
            embeds:  List of embed dicts built by ReportGenerator
            summary: Optional short text sent as `content` before the embeds
        """
        ok = True
        if summary:
            ok = self.send_text(summary[:2000]) and ok
        if embeds:
            ok = self.send_embeds(embeds) and ok
        return ok

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _post(self, payload: Dict[str, Any]) -> bool:
        """POST payload to primary webhook, then fallback, then file log."""
        payload.setdefault("username", self._username)
        if self._avatar_url:
            payload.setdefault("avatar_url", self._avatar_url)

        for url, label in [(self._primary, "primary"), (self._fallback, "fallback")]:
            if not url:
                continue
            try:
                resp = requests.post(url, json=payload, timeout=self._timeout)
                # 204 No Content = success for Discord webhooks
                if resp.status_code in (200, 204):
                    logger.info(f"[Discord] Message sent via {label} webhook")
                    return True
                logger.warning(
                    f"[Discord] {label} webhook returned HTTP {resp.status_code}: {resp.text[:200]}"
                )
            except requests.exceptions.ConnectionError as exc:
                logger.warning(f"[Discord] {label} webhook connection error: {exc}")
            except Exception as exc:
                logger.warning(f"[Discord] {label} webhook error: {exc}")

        # Both webhooks failed — write to fallback log
        logger.error("[Discord] Both webhooks failed — writing to fallback log")
        self._write_fallback(payload)
        return False

    def _write_fallback(self, payload: Dict[str, Any]) -> None:
        """Persist failed message to local log file."""
        import json
        try:
            FALLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            with FALLBACK_LOG.open("a", encoding="utf-8") as f:
                f.write(f"\n[{ts}]\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n{'─'*60}\n")
        except Exception as exc:
            logger.error(f"[Discord] Could not write fallback log: {exc}")


# ── Embed builders (helpers for ReportGenerator) ──────────────────────────────

def regime_embed(regime: Optional[Dict[str, Any]], market: str) -> Dict[str, Any]:
    """Build a Discord embed dict for a regime detection result."""
    if not regime:
        return {
            "title": f"{market} Market Regime",
            "description": "*No data available*",
            "color": COLOR_GREY,
        }

    score = regime.get("score", 0)
    label = regime.get("regime", "unknown")
    as_of = regime.get("as_of", "?")
    signals = regime.get("signals", {})
    err = regime.get("error")

    # Colour based on score
    if score >= 3:
        color = COLOR_GREEN
    elif score >= 0:
        color = COLOR_YELLOW
    else:
        color = COLOR_RED

    score_bar = "▲" * max(score, 0) + "▼" * max(-score, 0)
    desc_lines = [
        f"**{label}**",
        f"Score: `{score:+d}/6`  {score_bar}",
        f"As of: {as_of}",
    ]
    if err:
        desc_lines.append(f"\n⚠️ {err[:120]}")

    fields = []
    if signals:
        bullish = [k.replace("_", " ") for k, v in signals.items() if v == 1]
        bearish = [k.replace("_", " ") for k, v in signals.items() if v == -1]
        if bullish:
            fields.append({"name": "✅ Bullish signals", "value": "\n".join(bullish), "inline": True})
        if bearish:
            fields.append({"name": "❌ Bearish signals", "value": "\n".join(bearish), "inline": True})

    embed: Dict[str, Any] = {
        "title": f"{'🇺🇸' if market == 'US' else '🇰🇷'} {market} Market Regime",
        "description": "\n".join(desc_lines),
        "color": color,
    }
    if fields:
        embed["fields"] = fields
    return embed


def valuation_embed(timing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a Discord embed dict for market valuation."""
    if not timing:
        return {"title": "💰 Market Valuation", "description": "*No data*", "color": COLOR_GREY}

    fields = []
    for mkt in ["us", "kr"]:
        d = timing.get(mkt, {})
        if not d:
            continue
        label = d.get("label", "N/A")
        signal = d.get("valuation_signal", 0)
        pe = d.get("pe_ratio")
        spread = d.get("fed_model_spread_pp")
        icon = "🟢" if signal == 1 else "🔴" if signal == -1 else "🟡"
        val_lines = [f"{icon} **{label}**"]
        if pe:
            val_lines.append(f"P/E: {pe:.1f}x")
        if spread is not None:
            val_lines.append(f"Earnings yield spread: {spread:+.1f}pp")
        fields.append({"name": mkt.upper(), "value": "\n".join(val_lines), "inline": True})

    return {
        "title": "💰 Market Valuation",
        "color": COLOR_BLUE,
        "fields": fields or [{"name": "Status", "value": "No data", "inline": False}],
    }


def decision_embed(
    decision: Optional[Dict[str, Any]],
    market: str,
    max_stocks: int = 5,
) -> Dict[str, Any]:
    """Build a Discord embed dict for a decision result."""
    icon = "📈" if market == "US" else "🏢"
    title = f"{icon} {market} Top Stocks"

    if not decision:
        return {"title": title, "description": "*No data*", "color": COLOR_GREY}

    regime = decision.get("economic_regime", "unknown")
    rec = decision.get("recommendation", "")
    top = decision.get("top_stocks", [])[:max_stocks]
    favorable = decision.get("equities_favorable", False)

    color = COLOR_GREEN if favorable else COLOR_RED
    desc = f"Regime: `{regime}`\n{rec[:120]}"

    fields = []
    if top:
        stock_lines = []
        for s in top:
            sid = s.get("stock_id", "?")
            kcode = s.get("kiwoom_code", "")
            score = s.get("score", 0.0)
            reasons = ", ".join(s.get("reasons", [])[:2]) or "—"
            code_str = f" ({kcode})" if kcode else ""
            stock_lines.append(f"`{sid}`{code_str}  {score:+.3f}  {reasons}")
        fields.append({
            "name": "Top picks",
            "value": "\n".join(stock_lines),
            "inline": False,
        })

    if decision.get("alternatives"):
        fields.append({
            "name": "Alternatives",
            "value": "\n".join(f"• {a}" for a in decision["alternatives"][:4]),
            "inline": False,
        })

    embed: Dict[str, Any] = {"title": title, "description": desc, "color": color}
    if fields:
        embed["fields"] = fields
    return embed


def orders_embed(order_log: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Build a Discord embed dict for the order log."""
    if not order_log:
        return {
            "title": "📋 Orders This Cycle",
            "description": "*No orders placed*",
            "color": COLOR_GREY,
        }

    lines = []
    for o in order_log[:10]:
        side = o.get("side", "?").upper()
        ticker = o.get("ticker") or o.get("stock_code") or o.get("currency", "?")
        qty = o.get("quantity", "?")
        price = o.get("price")
        status = o.get("status", "?")
        paper = " [PAPER]" if o.get("paper_trading") else ""
        price_str = f"@ {price:,.0f} KRW" if price else "(market)"
        err = f" ⚠️ {o['error'][:40]}" if o.get("error") else ""
        lines.append(f"`{side}` **{ticker}** ×{qty} {price_str} → `{status}`{paper}{err}")

    color = COLOR_YELLOW if any(o.get("paper_trading") for o in order_log) else COLOR_GREEN
    return {
        "title": "📋 Orders This Cycle",
        "description": "\n".join(lines),
        "color": color,
    }


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    notifier = DiscordWebhookNotifier()
    ok = notifier.send_embed(
        title="🧪 Stock_ML Test",
        description="Discord webhook notifier is working correctly.",
        color=COLOR_GREEN,
        fields=[{"name": "Status", "value": "✅ Connected", "inline": True}],
        footer="Stock_ML automated test",
    )
    print("Sent:", ok)
