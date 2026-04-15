"""
KR Order Executor
=================
Bridges KRDecisionMaker output → KiwoomBroker order placement.

This is the only place where a decision becomes a real (or paper) trade.

Signal logic:
  - BUY:  equities_favorable=True AND stock score in top N AND score > min_score_threshold
  - SELL: equities_favorable=False (regime turned unfavorable) → close all positions
  - HOLD: no action if signals are weak or regime is unclear

Position sizing:
  - Default: 1 share per signal (paper-trading safe)
  - With budget: allocate proportionally to score within a fixed KRW budget

Safety controls (ALWAYS enforced regardless of paper_trading flag):
  - Max N signals per cycle (default 3) — prevents overtrading
  - Min score threshold — ignores low-conviction signals
  - Never short — only long (buy) or flat (sell to close)
  - All executed orders are logged and returned for the reporter

paper_trading (default True):
  - When True: KiwoomBroker logs the order but does NOT call the real API.
  - Must be explicitly set to False to send live orders.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from brokers.base_broker import BaseBroker, OrderResult

logger = logging.getLogger(__name__)


class KROrderExecutor:
    """
    Executes Korean stock orders based on KRDecisionMaker output.

    Args:
        broker:              A KiwoomBroker (or any BaseBroker) instance.
        max_signals:         Max number of buy orders per cycle (default 3).
        min_score_threshold: Minimum score to generate a buy signal (default 0.01).
        default_quantity:    Shares to buy per signal when no budget given (default 1).
        budget_krw:          Optional KRW budget to split across signals.
                             If set, uses proportional allocation.
    """

    def __init__(
        self,
        broker: BaseBroker,
        max_signals: int = 3,
        min_score_threshold: float = 0.01,
        default_quantity: int = 1,
        budget_krw: Optional[float] = None,
    ):
        self.broker = broker
        self.max_signals = max_signals
        self.min_score_threshold = min_score_threshold
        self.default_quantity = default_quantity
        self.budget_krw = budget_krw
        self._session_orders: List[OrderResult] = []

    # ── Main execution entry ───────────────────────────────────────────────────

    def execute(self, kr_decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate and execute orders from a KRDecisionMaker decision dict.

        Args:
            kr_decision: Output of KRDecisionMaker.run()

        Returns:
            List of OrderResult.to_dict() for each order attempted.
        """
        orders: List[OrderResult] = []

        regime = kr_decision.get("economic_regime", "unknown")
        favorable = kr_decision.get("equities_favorable", False)
        top_stocks = kr_decision.get("top_stocks", [])

        logger.info(
            f"[KROrderExecutor] regime={regime} favorable={favorable} "
            f"candidates={len(top_stocks)} paper={self.broker.paper_trading}"
        )

        if favorable:
            orders = self._execute_buy_signals(top_stocks, kr_decision)
        else:
            orders = self._execute_sell_signals(kr_decision)

        self._session_orders.extend(orders)
        order_dicts = [o.to_dict() for o in orders]

        self._log_summary(order_dicts, regime)
        return order_dicts

    # ── Signal execution ───────────────────────────────────────────────────────

    def _execute_buy_signals(
        self, top_stocks: List[Dict], decision: Dict
    ) -> List[OrderResult]:
        """Place buy orders for top-ranked stocks above min_score_threshold."""
        results: List[OrderResult] = []

        # Filter to high-conviction signals only
        signals = [
            s for s in top_stocks
            if s.get("score", 0) >= self.min_score_threshold
            and s.get("kiwoom_code")  # must have a valid Kiwoom code
        ][:self.max_signals]

        if not signals:
            logger.info(
                f"[KROrderExecutor] No buy signals above threshold "
                f"(min_score={self.min_score_threshold})"
            )
            return []

        # Optionally fetch live prices for budget allocation
        prices = self._fetch_prices(signals)

        for stock in signals:
            ticker = stock["kiwoom_code"]
            qty = self._calc_quantity(stock, signals, prices)

            if qty <= 0:
                logger.info(f"[KROrderExecutor] Skipping {ticker} — qty={qty}")
                continue

            result = self.broker.place_order(
                ticker=ticker,
                side="buy",
                quantity=qty,
                price=None,  # market order
            )
            result.raw_response = result.raw_response or {}
            result.raw_response["stock_id"] = stock.get("stock_id", ticker)
            result.raw_response["score"] = stock.get("score")
            results.append(result)

            if not result.success:
                logger.error(f"[KROrderExecutor] BUY failed for {ticker}: {result.error}")

        return results

    def _execute_sell_signals(self, decision: Dict) -> List[OrderResult]:
        """
        When regime turns unfavorable, sell all current positions.
        Fetches positions from broker then places sell orders.
        """
        results: List[OrderResult] = []

        try:
            positions = self.broker.get_positions()
        except Exception as exc:
            logger.error(f"[KROrderExecutor] get_positions failed: {exc}")
            return []

        if not positions:
            logger.info("[KROrderExecutor] No open positions to sell")
            return []

        for pos in positions:
            # Kiwoom position keys vary; try common field names
            ticker = pos.get("stk_cd") or pos.get("stock_code") or pos.get("ticker")
            qty_raw = pos.get("hldg_qty") or pos.get("quantity") or pos.get("qty", 0)
            qty = int(float(str(qty_raw).replace(",", ""))) if qty_raw else 0

            if not ticker or qty <= 0:
                continue

            result = self.broker.place_order(
                ticker=ticker,
                side="sell",
                quantity=qty,
                price=None,  # market order
            )
            results.append(result)

            if not result.success:
                logger.error(f"[KROrderExecutor] SELL failed for {ticker}: {result.error}")

        return results

    # ── Position sizing ────────────────────────────────────────────────────────

    def _calc_quantity(
        self,
        stock: Dict,
        all_signals: List[Dict],
        prices: Dict[str, float],
    ) -> int:
        """
        Calculate order quantity.

        - With budget_krw: allocate proportionally to score, capped by current price.
        - Without budget: use default_quantity (1 share).
        """
        ticker = stock.get("kiwoom_code", "")

        if self.budget_krw is None or not prices:
            return self.default_quantity

        current_price = prices.get(ticker)
        if not current_price or current_price <= 0:
            logger.warning(f"[KROrderExecutor] No price for {ticker} — using default qty")
            return self.default_quantity

        # Proportional allocation by score (normalised to sum=1)
        total_score = sum(max(s.get("score", 0), 0) for s in all_signals)
        if total_score <= 0:
            weight = 1.0 / len(all_signals)
        else:
            weight = max(stock.get("score", 0), 0) / total_score

        allocated_krw = self.budget_krw * weight
        qty = int(allocated_krw / current_price)
        return max(qty, 0)

    def _fetch_prices(self, signals: List[Dict]) -> Dict[str, float]:
        """
        Try to fetch current prices for all tickers using KiwoomBroker.get_current_price().
        Falls back to empty dict if the broker doesn't support it or errors out.
        """
        prices: Dict[str, float] = {}
        if not self.budget_krw:
            return prices  # no budget = no need for prices

        get_price = getattr(self.broker, "get_current_price", None)
        if not callable(get_price):
            return prices

        for stock in signals:
            ticker = stock.get("kiwoom_code", "")
            if not ticker:
                continue
            try:
                price = get_price(ticker)
                if price and price > 0:
                    prices[ticker] = price
            except Exception as exc:
                logger.warning(f"[KROrderExecutor] price fetch failed ({ticker}): {exc}")

        return prices

    # ── Logging / reporting ────────────────────────────────────────────────────

    def _log_summary(self, orders: List[Dict], regime: str) -> None:
        if not orders:
            logger.info(f"[KROrderExecutor] No orders placed (regime={regime})")
            return
        successes = sum(1 for o in orders if o.get("success"))
        logger.info(
            f"[KROrderExecutor] {len(orders)} order(s) attempted, "
            f"{successes} succeeded (regime={regime}, paper={self.broker.paper_trading})"
        )
        for o in orders:
            status = o.get("status", "?")
            err = f" — {o['error']}" if o.get("error") else ""
            logger.info(
                f"  {o['side'].upper():4s} {o['ticker']:10s} x{o['quantity']} "
                f"[{status}]{err}"
            )

    def get_session_orders(self) -> List[Dict[str, Any]]:
        """Return all orders placed in this session as dicts."""
        return [o.to_dict() for o in self._session_orders]

    def clear_session(self) -> None:
        self._session_orders.clear()
