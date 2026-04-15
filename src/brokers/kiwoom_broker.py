"""
Kiwoom REST API Broker
======================
Production-ready client for the Kiwoom Open API (REST).

Implements BaseBroker for consistent interface across all brokers.

Responsibilities:
  - OAuth2 token management with automatic refresh
  - Account balance / position queries
  - Stock order placement (buy / sell, market or limit)
  - Market data fetch (current price, orderbook) via Kiwoom REST
  - paper_trading mode: logs orders without sending real requests

Required environment variables (.env):
    KIWOOM_APP_KEY      — OAuth2 app key
    KIWOOM_SECRET_KEY   — OAuth2 secret key

API reference: https://openapi.kiwoom.com/guide/apiguide?dummyVal=0

Kiwoom REST endpoints used:
    POST /oauth2/token          — auth (TR: au10001)
    POST /api/dostk/ordr        — orders (TR: kt10000 buy / kt10001 sell)
    GET  /api/dostk/acnt        — account (TR: kt00018 balance / kt00019 positions)
    GET  /api/dostk/mrkcond     — market data (TR: kt10001 current price)

NOTE: This module routes orders for Korean stocks (KRX/KOSPI/KOSDAQ) ONLY.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from brokers.base_broker import BaseBroker, OrderResult

load_dotenv()

logger = logging.getLogger(__name__)

KIWOOM_HOST = "https://api.kiwoom.com"

# Retry settings for Kiwoom API calls
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.5   # seconds, multiplied per attempt


# ── Token Manager ─────────────────────────────────────────────────────────────

class _TokenManager:
    """Manages OAuth2 access token with automatic refresh before expiry."""

    REFRESH_BUFFER_SEC = 300  # refresh 5 min before expiry

    def __init__(self, app_key: str, secret_key: str):
        self._app_key = app_key
        self._secret_key = secret_key
        self._token: Optional[str] = None
        self._expires_at: Optional[datetime] = None

    def _fetch(self) -> None:
        url = KIWOOM_HOST + "/oauth2/token"
        body = {
            "grant_type": "client_credentials",
            "appkey": self._app_key,
            "secretkey": self._secret_key,
        }
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.post(
                    url,
                    headers={"Content-Type": "application/json;charset=UTF-8"},
                    json=body,
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                self._token = data.get("token") or data.get("access_token")
                expires_in = int(data.get("expires_in", 86400))
                self._expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                logger.info(f"[Kiwoom] Token refreshed, expires {self._expires_at.isoformat()}")
                return
            except requests.RequestException as exc:
                wait = _RETRY_BACKOFF * (attempt + 1)
                logger.warning(f"[Kiwoom] Token fetch attempt {attempt+1} failed: {exc} — retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError("Kiwoom token fetch failed after all retries")

    def get(self) -> str:
        if self._token is None or self._expiring_soon():
            self._fetch()
        return self._token  # type: ignore[return-value]

    def _expiring_soon(self) -> bool:
        if self._expires_at is None:
            return True
        return (self._expires_at - datetime.utcnow()).total_seconds() < self.REFRESH_BUFFER_SEC


# ── Kiwoom Broker ─────────────────────────────────────────────────────────────

class KiwoomBroker(BaseBroker):
    """
    Kiwoom REST API broker for Korean stocks.

    Args:
        app_key:       Kiwoom OAuth2 app key (env KIWOOM_APP_KEY)
        secret_key:    Kiwoom OAuth2 secret key (env KIWOOM_SECRET_KEY)
        paper_trading: If True (default), log orders without real API calls.
                       Must be explicitly set to False to send live orders.

    Raises:
        ValueError: if app_key or secret_key are missing
    """

    def __init__(
        self,
        app_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper_trading: bool = True,
    ):
        super().__init__(paper_trading=paper_trading)

        self._app_key = app_key or os.getenv("KIWOOM_APP_KEY", "")
        self._secret_key = secret_key or os.getenv("KIWOOM_SECRET_KEY", "")

        if not self._app_key or not self._secret_key:
            raise ValueError(
                "KIWOOM_APP_KEY and KIWOOM_SECRET_KEY must be set in .env "
                "or passed as constructor arguments."
            )

        self._tokens = _TokenManager(self._app_key, self._secret_key)

        if paper_trading:
            logger.info("[Kiwoom] Initialised in PAPER TRADING mode — no real orders will be sent")
        else:
            logger.warning("[Kiwoom] Initialised in LIVE TRADING mode — real orders WILL be sent")

    # ── Internal HTTP helpers ──────────────────────────────────────────────────

    def _auth_headers(self, api_id: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {self._tokens.get()}",
            "api-id": api_id,
            "cont-yn": "N",
            "next-key": "",
        }

    def _post(self, endpoint: str, api_id: str, body: Dict) -> Dict:
        url = KIWOOM_HOST + endpoint
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.post(
                    url, headers=self._auth_headers(api_id), json=body, timeout=15
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                wait = _RETRY_BACKOFF * (attempt + 1)
                logger.warning(f"[Kiwoom] POST {endpoint} attempt {attempt+1} failed: {exc}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)
        raise RuntimeError(f"Kiwoom POST {endpoint} failed after {_MAX_RETRIES} attempts")

    def _get(self, endpoint: str, api_id: str, params: Dict) -> Dict:
        url = KIWOOM_HOST + endpoint
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(
                    url, headers=self._auth_headers(api_id), params=params, timeout=15
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                wait = _RETRY_BACKOFF * (attempt + 1)
                logger.warning(f"[Kiwoom] GET {endpoint} attempt {attempt+1} failed: {exc}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)
        raise RuntimeError(f"Kiwoom GET {endpoint} failed after {_MAX_RETRIES} attempts")

    # ── BaseBroker implementation ──────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        exchange: str = "KRX",
    ) -> OrderResult:
        """
        Place a Korean stock order via Kiwoom REST API.

        Args:
            ticker:   6-digit KRX stock code (e.g. "005930")
            side:     "buy" or "sell"
            quantity: Number of shares (must be integer for KRX)
            price:    Limit price in KRW; None = market order
            exchange: "KRX" (default) | "NXT" | "KSQ"

        Returns:
            OrderResult
        """
        is_buy = side.lower() == "buy"
        qty_int = int(quantity)

        # Kiwoom trade type codes
        if price is not None:
            trde_tp = "1" if is_buy else "2"   # limit
            ord_uv = str(int(price))
        else:
            trde_tp = "3" if is_buy else "4"   # market
            ord_uv = ""

        api_id = "kt10000" if is_buy else "kt10001"

        result = OrderResult(
            success=False,
            side=side,
            ticker=ticker,
            quantity=qty_int,
            price=price,
            paper_trading=self.paper_trading,
        )

        # ── Paper trading: log only ────────────────────────────────────────────
        if self.paper_trading:
            result.success = True
            result.status = "paper_trade"
            logger.info(
                f"[Kiwoom][PAPER] {side.upper()} {ticker} x{qty_int} "
                f"{'@'+str(price) if price else 'market'} on {exchange}"
            )
            self._record(result)
            return result

        # ── Live order ────────────────────────────────────────────────────────
        body = {
            "dmst_stex_tp": exchange,
            "stk_cd": ticker,
            "ord_qty": str(qty_int),
            "ord_uv": ord_uv,
            "trde_tp": trde_tp,
            "cond_uv": "",
        }
        try:
            resp = self._post("/api/dostk/ordr", api_id, body)
            result.success = True
            result.status = "submitted"
            result.order_id = resp.get("ord_no") or resp.get("output", {}).get("ord_no")
            result.raw_response = resp
            logger.info(
                f"[Kiwoom][LIVE] {side.upper()} {ticker} x{qty_int} submitted — "
                f"order_id={result.order_id}"
            )
        except Exception as exc:
            result.status = "error"
            result.error = str(exc)
            logger.error(f"[Kiwoom] Order failed ({ticker}): {exc}")

        self._record(result)
        return result

    def get_balance(self) -> Dict[str, Any]:
        """
        Query account cash balance (TR: kt00018).
        Returns raw Kiwoom response dict, or {"error": ...} on failure.
        """
        if self.paper_trading:
            logger.info("[Kiwoom][PAPER] get_balance called — returning mock data")
            return {"paper_trading": True, "cash_balance": "N/A"}
        try:
            return self._get("/api/dostk/acnt", "kt00018", {"qry_tp": "1"})
        except Exception as exc:
            logger.error(f"[Kiwoom] get_balance failed: {exc}")
            return {"error": str(exc)}

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Query current stock holdings (TR: kt00019).
        Returns list of position dicts, or [] on failure.
        """
        if self.paper_trading:
            logger.info("[Kiwoom][PAPER] get_positions called — returning empty list")
            return []
        try:
            resp = self._get("/api/dostk/acnt", "kt00019", {"qry_tp": "2"})
            positions = resp.get("output", resp.get("data", []))
            return positions if isinstance(positions, list) else []
        except Exception as exc:
            logger.error(f"[Kiwoom] get_positions failed: {exc}")
            return []

    # ── Market data (Kiwoom REST) ──────────────────────────────────────────────

    def get_current_price(self, stock_code: str) -> Optional[float]:
        """
        Fetch the current price of a Korean stock via Kiwoom REST API.

        Uses TR: kt10001 (주식 현재가 조회).
        Falls back gracefully to None on any error.

        Args:
            stock_code: 6-digit KRX code, e.g. "005930"

        Returns:
            Current price as float (KRW), or None if unavailable.
        """
        if self.paper_trading:
            # In paper trading mode we don't need a live token to get prices,
            # but it's still safe to call this for position sizing.
            # If token fetch fails, just return None.
            pass
        try:
            resp = self._get(
                endpoint="/api/dostk/mrkcond",
                api_id="kt10001",
                params={"stk_cd": stock_code},
            )
            # Kiwoom returns current price in "output" → "stck_prpr" (주식 현재가)
            output = resp.get("output", {})
            price_str = output.get("stck_prpr") or output.get("cur_prc")
            if price_str:
                # Kiwoom sometimes returns price with commas or as string
                return float(str(price_str).replace(",", ""))
        except Exception as exc:
            logger.warning(f"[Kiwoom] get_current_price({stock_code}) failed: {exc}")
        return None

    def get_orderbook(self, stock_code: str) -> Dict[str, Any]:
        """
        Fetch level-1 orderbook for a Korean stock.

        Uses TR: kt10004 (주식 호가 조회).

        Args:
            stock_code: 6-digit KRX code

        Returns:
            Dict with "bids" and "asks" lists, or empty dict on failure.
        """
        try:
            resp = self._get(
                endpoint="/api/dostk/mrkcond",
                api_id="kt10004",
                params={"stk_cd": stock_code},
            )
            return resp
        except Exception as exc:
            logger.warning(f"[Kiwoom] get_orderbook({stock_code}) failed: {exc}")
            return {}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    broker = KiwoomBroker(paper_trading=True)

    # Paper trade: Samsung Electronics
    result = broker.buy(ticker="005930", quantity=1)
    print("Buy result:", result.to_dict())

    result = broker.sell(ticker="005930", quantity=1)
    print("Sell result:", result.to_dict())

    print("Order log:", broker.get_order_log())
