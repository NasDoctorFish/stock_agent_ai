"""
Telegram Sender
===============
Sends messages to a Telegram chat using the Bot API.
Uses only the standard `requests` library — no extra dependencies.

Required environment variables (.env):
    TELEGRAM_BOT_KEY    — Bot token from @BotFather (e.g. "123456:ABC-DEF...")
    TELEGRAM_CHAT_ID    — Numeric chat ID to send messages to.
                          HOW TO GET IT:
                            1. Start a conversation with your bot on Telegram.
                            2. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates
                            3. Find "chat": {"id": <NUMBER>} in the response.
                            4. Add that number to .env as TELEGRAM_CHAT_ID.

Optional:
    TELEGRAM_PARSE_MODE — "Markdown" or "HTML" (default: "Markdown")

If Telegram fails (network error, missing credentials), the message is
written to a fallback file at logs/telegram_fallback.log so nothing is lost.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
FALLBACK_LOG = Path(__file__).resolve().parents[2] / "logs" / "telegram_fallback.log"


class TelegramSender:
    """
    Sends messages to a Telegram chat.

    Args:
        bot_token:  Bot token (defaults to env TELEGRAM_BOT_KEY)
        chat_id:    Target chat ID (defaults to env TELEGRAM_CHAT_ID)
        parse_mode: "Markdown" or "HTML" (default "Markdown")
        timeout:    HTTP request timeout in seconds (default 10)
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        parse_mode: str = "Markdown",
        timeout: int = 10,
    ):
        self._token = bot_token or os.getenv("TELEGRAM_BOT_KEY", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._parse_mode = parse_mode
        self._timeout = timeout

        if not self._token:
            logger.warning("TELEGRAM_BOT_KEY not set — messages will only be written to fallback log")
        if not self._chat_id:
            logger.warning("TELEGRAM_CHAT_ID not set — messages will only be written to fallback log")

    def _is_configured(self) -> bool:
        return bool(self._token and self._chat_id)

    def send(self, text: str) -> bool:
        """
        Send a text message.

        Returns True on success, False on failure.
        On failure the message is always saved to the fallback log.
        """
        if not self._is_configured():
            logger.warning("Telegram not configured; writing to fallback log only")
            self._write_fallback(text)
            return False

        url = f"{TELEGRAM_API_BASE}/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": self._parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                raise RuntimeError(f"Telegram API error: {data.get('description')}")
            logger.info("Telegram message sent successfully")
            return True
        except Exception as exc:
            logger.error(f"Telegram send failed: {exc} — saving to fallback log")
            self._write_fallback(text)
            return False

    def send_chunked(self, text: str, max_chars: int = 4096) -> bool:
        """
        Split and send a long message in chunks (Telegram limit is 4096 chars).
        Returns True only if all chunks succeeded.
        """
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
        return all(self.send(chunk) for chunk in chunks)

    def _write_fallback(self, text: str) -> None:
        """Write message to fallback log file."""
        try:
            FALLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
            with FALLBACK_LOG.open("a", encoding="utf-8") as f:
                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                f.write(f"\n[{ts}]\n{text}\n{'─'*60}\n")
        except Exception as exc:
            logger.error(f"Could not write to fallback log: {exc}")


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sender = TelegramSender()
    ok = sender.send("*Test message* from Stock\\_ML Telegram Sender ✅")
    print("Sent:", ok)
