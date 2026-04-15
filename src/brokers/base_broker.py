"""
BaseBroker — Abstract Broker Interface
=======================================
All broker implementations (KiwoomBroker, CoinoneBroker, future US broker)
must inherit from this class and implement the abstract methods.

paper_trading mode:
    When paper_trading=True the broker MUST log all orders without sending
    real API requests.  This is the safe default for all new integrations.

OrderResult:
    Standardised return type for every order operation.  Callers can always
    check `result.success` without caring about broker-specific response formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class OrderResult:
    """Standardised result for any order operation."""
    success: bool
    side: str                           # "buy" | "sell"
    ticker: str                         # broker-native code (e.g. "005930", "BTC")
    quantity: float
    price: Optional[float]              # None = market order
    paper_trading: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    order_id: Optional[str] = None      # broker-assigned order ID if available
    status: str = "pending"             # "dry_run" | "submitted" | "filled" | "error"
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "side": self.side,
            "ticker": self.ticker,
            "quantity": self.quantity,
            "price": self.price,
            "paper_trading": self.paper_trading,
            "timestamp": self.timestamp,
            "order_id": self.order_id,
            "status": self.status,
            "error": self.error,
        }


class BaseBroker(ABC):
    """
    Abstract base class for all brokers.

    Subclasses must implement:
        place_order(ticker, side, quantity, price) -> OrderResult
        get_balance() -> dict
        get_positions() -> list[dict]

    paper_trading:
        Default True.  Must be explicitly set to False by the caller to send
        real orders.  Any subclass that ignores this flag is a bug.
    """

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self._order_log: List[OrderResult] = []

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def place_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> OrderResult:
        """
        Place a buy or sell order.

        Args:
            ticker:   Broker-native instrument code
            side:     "buy" or "sell"
            quantity: Number of units (shares / coins)
            price:    Limit price; None = market order

        Returns:
            OrderResult with status and broker response.
        """

    @abstractmethod
    def get_balance(self) -> Dict[str, Any]:
        """Return current account cash/asset balance as a dict."""

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Return list of current open positions."""

    # ── Convenience shortcuts ──────────────────────────────────────────────────

    def buy(self, ticker: str, quantity: float, price: Optional[float] = None) -> OrderResult:
        return self.place_order(ticker, "buy", quantity, price)

    def sell(self, ticker: str, quantity: float, price: Optional[float] = None) -> OrderResult:
        return self.place_order(ticker, "sell", quantity, price)

    # ── Order log ──────────────────────────────────────────────────────────────

    def _record(self, result: OrderResult) -> None:
        """Append an OrderResult to the in-memory order log."""
        self._order_log.append(result)

    def get_order_log(self) -> List[Dict[str, Any]]:
        """Return a snapshot of all orders placed this session."""
        return [r.to_dict() for r in self._order_log]

    def clear_order_log(self) -> None:
        self._order_log.clear()
