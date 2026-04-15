"""
Coinone Broker
==============
Full implementation of the Coinone v2 REST API for crypto trading (KRW pairs).

Authentication (Coinone v2):
  1. Build payload dict including `access_token` and `nonce` (unix ms)
  2. JSON-encode → UTF-8 bytes → Base64-encode → encoded_payload string
  3. HMAC-SHA512 over encoded_payload using secret_key.upper() as the key
  4. Send as POST with headers:
       X-COINONE-PAYLOAD   : encoded_payload
       X-COINONE-SIGNATURE : hmac_hex

Environment variables (.env):
    COINONE_ACCESS_KEY  — API access key (used as `access_token` in every payload)
    COINONE_SECRET      — Secret for HMAC signing

paper_trading (default True):
    When True, all trade methods log the intent and return a paper OrderResult
    without touching the Coinone API.  Set False only for live trading.

Supported markets:
    KRW pairs: BTC, ETH, XRP, ADA, SOL, DOGE, etc.
    Coinone only supports KRW base; no USD or BTC cross-pairs.

NOTE: This broker handles CRYPTO only.
    Korean stocks → KiwoomBroker
    US stocks     → (not yet implemented)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from brokers.base_broker import BaseBroker, OrderResult

load_dotenv()

logger = logging.getLogger(__name__)

COINONE_API_BASE = "https://api.coinone.co.kr"

_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0   # seconds per retry attempt


class CoinoneBroker(BaseBroker):
    """
    Coinone REST API v2 broker for Korean crypto (KRW pairs).

    Args:
        access_key:    Coinone access key (env COINONE_ACCESS_KEY)
        secret:        Coinone HMAC secret (env COINONE_SECRET)
        paper_trading: True (default) = log only; False = live orders
    """

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret: Optional[str] = None,
        paper_trading: bool = True,
    ):
        super().__init__(paper_trading=paper_trading)

        self._access_key = access_key or os.getenv("COINONE_ACCESS_KEY", "").strip()
        self._secret = secret or os.getenv("COINONE_SECRET", "").strip()

        if not self._access_key or not self._secret:
            logger.warning(
                "[Coinone] COINONE_ACCESS_KEY or COINONE_SECRET not set — "
                "authenticated calls will fail"
            )

        if paper_trading:
            logger.info("[Coinone] Initialised in PAPER TRADING mode")
        else:
            logger.warning("[Coinone] Initialised in LIVE TRADING mode — real orders WILL be sent")

    # ── Authentication ─────────────────────────────────────────────────────────

    def _build_payload(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Add access_token and nonce to any payload dict."""
        payload = dict(extra)
        payload["access_token"] = self._access_key
        payload["nonce"] = int(time.time() * 1000)
        return payload

    def _sign(self, payload: Dict[str, Any]) -> tuple[str, str]:
        """
        Sign a payload using Coinone v2 HMAC-SHA512 scheme.

        Returns:
            (encoded_payload_str, signature_hex)
        """
        payload_json = json.dumps(payload, separators=(",", ":"))
        encoded = base64.b64encode(payload_json.encode("utf-8"))
        # Coinone requires the secret uppercased as the HMAC key
        sig = hmac.new(
            self._secret.upper().encode("utf-8"),
            encoded,
            hashlib.sha512,
        ).hexdigest()
        return encoded.decode("utf-8"), sig

    def _auth_headers(self, encoded_payload: str, signature: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-COINONE-PAYLOAD": encoded_payload,
            "X-COINONE-SIGNATURE": signature,
        }

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    def _post_private(self, endpoint: str, extra_payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to a private (authenticated) Coinone endpoint."""
        payload = self._build_payload(extra_payload)
        encoded, sig = self._sign(payload)
        headers = self._auth_headers(encoded, sig)
        url = COINONE_API_BASE + endpoint

        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if data.get("result") == "error":
                    raise RuntimeError(
                        f"Coinone API error at {endpoint}: "
                        f"[{data.get('errorCode')}] {data.get('errorMessage', data)}"
                    )
                return data
            except requests.RequestException as exc:
                wait = _RETRY_BACKOFF * (attempt + 1)
                logger.warning(f"[Coinone] POST {endpoint} attempt {attempt+1} failed: {exc}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)

        raise RuntimeError(f"Coinone POST {endpoint} failed after {_MAX_RETRIES} attempts")

    def _get_public(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """GET from a public (unauthenticated) Coinone endpoint."""
        url = COINONE_API_BASE + endpoint
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                wait = _RETRY_BACKOFF * (attempt + 1)
                logger.warning(f"[Coinone] GET {endpoint} attempt {attempt+1} failed: {exc}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)
        raise RuntimeError(f"Coinone GET {endpoint} failed after {_MAX_RETRIES} attempts")

    # ── BaseBroker implementation ──────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> OrderResult:
        """
        Place a buy or sell order on Coinone.

        Args:
            ticker:   Crypto symbol, e.g. "BTC", "ETH", "XRP"
            side:     "buy" or "sell"
            quantity: Coin quantity (float)
            price:    Limit price in KRW; None = market order

        Returns:
            OrderResult with Coinone order_id on success.
        """
        is_buy = side.lower() == "buy"
        currency = ticker.upper()

        result = OrderResult(
            success=False,
            side=side,
            ticker=currency,
            quantity=quantity,
            price=price,
            paper_trading=self.paper_trading,
        )

        # ── Paper mode ────────────────────────────────────────────────────────
        if self.paper_trading:
            result.success = True
            result.status = "paper_trade"
            order_type = "limit" if price else "market"
            logger.info(
                f"[Coinone][PAPER] {side.upper()} {quantity} {currency} "
                f"{'@ '+str(price)+' KRW' if price else '(market)'}  ({order_type})"
            )
            self._record(result)
            return result

        # ── Live order ────────────────────────────────────────────────────────
        try:
            if price is not None:
                # Limit order
                endpoint = "/v2/order/limit_buy/" if is_buy else "/v2/order/limit_sell/"
                data = self._post_private(endpoint, {
                    "price": str(int(price)),
                    "qty": str(quantity),
                    "currency": currency,
                })
            else:
                # Market order
                endpoint = "/v2/order/market_buy/" if is_buy else "/v2/order/market_sell/"
                field = "qty" if not is_buy else "price"   # market buy needs KRW amount
                value = str(quantity) if not is_buy else str(int(quantity))
                data = self._post_private(endpoint, {
                    field: value,
                    "currency": currency,
                })

            result.success = True
            result.status = "submitted"
            result.order_id = data.get("orderId") or data.get("order_id")
            result.raw_response = data
            logger.info(
                f"[Coinone][LIVE] {side.upper()} {quantity} {currency} submitted — "
                f"order_id={result.order_id}"
            )
        except Exception as exc:
            result.status = "error"
            result.error = str(exc)
            logger.error(f"[Coinone] Order failed ({currency}): {exc}")

        self._record(result)
        return result

    def get_balance(self) -> Dict[str, Any]:
        """
        Query all KRW and crypto balances.

        Returns:
            Dict with currency keys, each containing 'avail' and 'balance'.
            In paper mode returns a mock structure.

        Example response:
            {
                "krw": {"avail": "1000000", "balance": "1000000"},
                "btc": {"avail": "0.5", "balance": "0.5"},
                ...
            }
        """
        if self.paper_trading:
            logger.info("[Coinone][PAPER] get_balance — returning mock structure")
            return {"paper_trading": True, "krw": {"avail": "N/A", "balance": "N/A"}}
        try:
            data = self._post_private("/v2/account/balance/", {})
            # Remove metadata keys from the response
            return {k: v for k, v in data.items() if k not in ("result", "errorCode", "normalTime")}
        except Exception as exc:
            logger.error(f"[Coinone] get_balance failed: {exc}")
            return {"error": str(exc)}

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Return current non-zero crypto holdings as a list of position dicts.

        In live mode, calls get_balance() and filters to assets with balance > 0.
        """
        if self.paper_trading:
            return []
        try:
            balance = self.get_balance()
            positions = []
            for currency, info in balance.items():
                if currency in ("error", "paper_trading", "krw"):
                    continue
                avail = float(info.get("avail", 0))
                total = float(info.get("balance", 0))
                if total > 0:
                    positions.append({
                        "currency": currency.upper(),
                        "available": avail,
                        "total": total,
                    })
            return positions
        except Exception as exc:
            logger.error(f"[Coinone] get_positions failed: {exc}")
            return []

    # ── Public market data (no auth) ───────────────────────────────────────────

    def get_ticker(self, currency: str = "BTC") -> Dict[str, Any]:
        """
        Fetch the latest ticker for a KRW pair.

        Args:
            currency: Crypto symbol (e.g. "BTC", "ETH")

        Returns:
            Dict with keys: last, high, low, volume, timestamp, currency
        """
        try:
            data = self._get_public("/v2/ticker/", {"currency": currency.upper()})
            return {
                "currency": currency.upper(),
                "last": float(data.get("last", 0)),
                "high": float(data.get("high", 0)),
                "low": float(data.get("low", 0)),
                "volume": float(data.get("volume", 0)),
                "timestamp": data.get("timestamp"),
            }
        except Exception as exc:
            logger.error(f"[Coinone] get_ticker({currency}) failed: {exc}")
            return {"currency": currency.upper(), "error": str(exc)}

    def get_orderbook(self, currency: str = "BTC") -> Dict[str, Any]:
        """
        Fetch the current orderbook for a KRW pair.

        Returns:
            Dict with "bids" and "asks" lists (each item: {"price": float, "qty": float})
        """
        try:
            data = self._get_public("/v2/orderbook/", {"currency": currency.upper()})
            return {
                "currency": currency.upper(),
                "bids": [
                    {"price": float(b["price"]), "qty": float(b["qty"])}
                    for b in data.get("bid", [])
                ],
                "asks": [
                    {"price": float(a["price"]), "qty": float(a["qty"])}
                    for a in data.get("ask", [])
                ],
            }
        except Exception as exc:
            logger.error(f"[Coinone] get_orderbook({currency}) failed: {exc}")
            return {"currency": currency.upper(), "bids": [], "asks": []}

    def get_open_orders(self, currency: str = "BTC") -> List[Dict[str, Any]]:
        """
        List open (unfilled) orders for a currency.

        Returns:
            List of order dicts, or empty list on error / paper mode.
        """
        if self.paper_trading:
            return []
        try:
            data = self._post_private(
                "/v2/order/limit_orders/",
                {"currency": currency.upper()},
            )
            return data.get("limitOrders", [])
        except Exception as exc:
            logger.error(f"[Coinone] get_open_orders({currency}) failed: {exc}")
            return []

    def cancel_order(
        self,
        order_id: str,
        currency: str,
        price: float,
        qty: float,
        is_ask: bool,
    ) -> Dict[str, Any]:
        """
        Cancel an existing limit order.

        Args:
            order_id: Order ID returned by place_order
            currency: Crypto symbol
            price:    Original order price (required by Coinone cancel API)
            qty:      Original order quantity
            is_ask:   True if it was a sell (ask) order, False if buy (bid)

        Returns:
            API response dict, or {"error": ...} on failure.
        """
        if self.paper_trading:
            logger.info(f"[Coinone][PAPER] cancel_order {order_id} ({currency})")
            return {"paper_trading": True, "order_id": order_id, "status": "cancelled"}
        try:
            return self._post_private("/v2/order/cancel/", {
                "order_id": order_id,
                "price": str(int(price)),
                "qty": str(qty),
                "is_ask": 1 if is_ask else 0,
                "currency": currency.upper(),
            })
        except Exception as exc:
            logger.error(f"[Coinone] cancel_order({order_id}) failed: {exc}")
            return {"error": str(exc)}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    broker = CoinoneBroker(paper_trading=True)

    # Public data (no auth needed)
    print("\n--- BTC Ticker ---")
    ticker = broker.get_ticker("BTC")
    print(ticker)

    # Paper trade
    print("\n--- Paper BUY 0.001 BTC ---")
    result = broker.buy(ticker="BTC", quantity=0.001)
    print(result.to_dict())

    print("\n--- Paper SELL 0.001 ETH ---")
    result = broker.sell(ticker="ETH", quantity=0.001, price=3_000_000)
    print(result.to_dict())

    print("\n--- Order Log ---")
    for o in broker.get_order_log():
        print(o)
