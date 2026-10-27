"""Anthropic calls. Only the compact digest is ever sent. The API key is never logged or returned."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from .digest import digest_json
from .schemas import Strategy
from .service import ROOT

DEFAULT_MODEL = "claude-sonnet-5-5"
GEMINI_MODELS = "gemini-3.5-flash,gemini-3.1-pro-preview"
STRATEGY_MAX_TOKENS = 2000
REVIEW_MAX_TOKENS = 1500

_SYSTEM = ("You are a quantitative portfolio strategist for a small long-only US/KR equity universe. "
           "You only see a compact statistics digest. Be concrete and conservative; never promise returns. "
           "equity_by_regime is the fraction of NAV in stocks per macro regime (rest is cash). "
           "Use sell_rules.planned_sell_date only if the user notes ask for a horizon.")


def load_env() -> None:
    """Load ANTHROPIC_API_KEY (and AI_MODELS) from repo-root .env into os.environ; never prints anything."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            if k.upper() == "GEMINI_API_KEY":  # .env spells it Gemini_API_KEY
                k = "GEMINI_API_KEY"
            if k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "AI_MODELS") and k not in os.environ:
                os.environ[k] = v.strip().strip("'\"")


load_env()

MODELS = os.environ.get(
    "AI_MODELS",
    ",".join([DEFAULT_MODEL, "claude-opus-4-5", "claude-haiku-4-5"]
             + (GEMINI_MODELS.split(",") if os.environ.get("GEMINI_API_KEY") else []))).split(",")


def ai_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _gemini(model: str, system: str, user: str, max_tokens: int, json_out: bool = False) -> str:
    """Gemini REST generateContent (no extra dependency). Key goes in a header, never the URL."""
    import httpx
    cfg = {"maxOutputTokens": max_tokens}
    if json_out:
        cfg["responseMimeType"] = "application/json"
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": cfg}
    for attempt in range(3):  # Gemini returns transient 503s under load
        r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                       headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=120.0, json=body)
        if r.status_code != 503:
            break
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()
    return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])


def get_client():
    """Overridable in tests (monkeypatch ai.get_client)."""
    import anthropic
    return anthropic.Anthropic(timeout=120.0, max_retries=1)


def _safe(e: Exception) -> str:
    msg = f"{type(e).__name__}: {e}"
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        k = os.environ.get(name, "")
        if k:
            msg = msg.replace(k, "***")
    return msg


def _tool_input(resp) -> dict:
    for b in resp.content:
        if getattr(b, "type", "") == "tool_use":
            return b.input
    raise ValueError("model returned no tool_use block")


def propose_strategy(digest: dict, params: dict, notes: str = "", model: Optional[str] = None,
                     full: bool = False) -> Strategy:
    ask = ("Design ONE complete strategy: top_n, rebalance schedule (and rebalance_days if custom), equity_by_regime "
           "allocation, sell rules incl. planned_sell_date if sensible, rebalance_triggers, risk_notes."
           if full else "Recommend ONE strategy fitting the current conditions.")
    user = (f"{ask}\nBacktest window: {params.get('start')} to {params.get('end')}; benchmark {params.get('benchmark')}.\n"
            f"User notes: {notes or 'none'}\nDIGEST:{digest_json(digest)}")
    try:
        if (model or "").startswith("gemini"):
            schema = json.dumps(Strategy.model_json_schema())
            s = Strategy.model_validate_json(_gemini(
                model, _SYSTEM, f"{user}\nRespond with ONLY a JSON object matching this JSON Schema:\n{schema}",
                STRATEGY_MAX_TOKENS * 2, json_out=True))
            return _with_id(s)
        resp = get_client().messages.create(
            model=model or DEFAULT_MODEL, max_tokens=STRATEGY_MAX_TOKENS, system=_SYSTEM,
            tools=[{"name": "submit_strategy", "description": "Submit the strategy.",
                    "input_schema": Strategy.model_json_schema()}],
            tool_choice={"type": "auto"},  # forced "tool"/"any" rejected by sonnet-5-5
            messages=[{"role": "user", "content": user + "\nRespond only by calling submit_strategy."}])
        s = Strategy.model_validate(_tool_input(resp))
    except Exception as e:
        raise RuntimeError(_safe(e)) from None
    return _with_id(s)


def _with_id(s: Strategy) -> Strategy:
    if not s.id:
        s.id = "ai-" + "".join(c if c.isalnum() else "-" for c in s.name.lower())[:40].strip("-")
    return s


def review(digest_with_backtest: dict, strategy: Strategy, model: Optional[str] = None) -> str:
    user = (f"Review this backtest vs benchmark in <=300 words markdown: verdict, what drove results by regime, "
            f"weaknesses, 2-3 concrete tweaks.\nSTRATEGY:{strategy.model_dump_json()}\nDIGEST:{digest_json(digest_with_backtest)}")
    try:
        if (model or "").startswith("gemini"):
            return _gemini(model, _SYSTEM, user, REVIEW_MAX_TOKENS * 2)
        resp = get_client().messages.create(model=model or DEFAULT_MODEL, max_tokens=REVIEW_MAX_TOKENS,
                                            system=_SYSTEM, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    except Exception as e:
        raise RuntimeError(_safe(e)) from None
