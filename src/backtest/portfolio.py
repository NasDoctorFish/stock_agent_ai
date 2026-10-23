"""
PortfolioSimulator
------------------
Tracks cash, equity positions, and NAV over time.
Applies slippage and commission on all trades.

No external API calls — operates entirely on data passed in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single executed trade."""
    date: pd.Timestamp
    stock_id: str
    action: str          # 'BUY' | 'SELL'
    shares: float
    price: float         # raw price before slippage
    exec_price: float    # price after slippage
    gross_value: float   # abs(shares * exec_price)
    commission: float
    net_cash_flow: float # negative for buys, positive for sells


class PortfolioSimulator:
    """
    Simulates a portfolio that holds cash and equity positions.

    Parameters
    ----------
    initial_capital : float
        Starting cash balance.
    slippage_pct : float
        Fraction added to buy price / subtracted from sell price.
        Default 0.001 (10 bps one-way).
    commission_pct : float
        Fraction of gross trade value charged as commission.
        Default 0.00015 (1.5 bps).
    """

    def __init__(
        self,
        initial_capital: float,
        slippage_pct: float = 0.001,
        commission_pct: float = 0.00015,
    ) -> None:
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct

        self.cash: float = initial_capital
        self.positions: Dict[str, float] = {}   # stock_id → shares held
        self._trade_log: List[dict] = []
        self._nav_history: List[dict] = []       # [{date, nav}]

    # ─── NAV ──────────────────────────────────────────────────────────────────

    def current_nav(self, prices: Dict[str, float]) -> float:
        """Compute current NAV = cash + sum(shares * price) for each position."""
        equity_value = sum(
            shares * prices.get(sid, 0.0)
            for sid, shares in self.positions.items()
        )
        return self.cash + equity_value

    def record_nav(self, date: pd.Timestamp, prices: Dict[str, float]) -> float:
        """Record today's NAV and append to history. Returns the NAV value."""
        nav = self.current_nav(prices)
        self._nav_history.append({"date": date, "nav": nav})
        return nav

    def get_nav_series(self) -> pd.Series:
        """Return NAV history as a pd.Series with DatetimeIndex."""
        if not self._nav_history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self._nav_history).set_index("date")
        df.index = pd.to_datetime(df.index)
        return df["nav"]

    # ─── Trading ─────────────────────────────────────────────────────────────

    def _execute_buy(
        self,
        date: pd.Timestamp,
        stock_id: str,
        shares: float,
        price: float,
    ) -> Trade:
        """Buy `shares` of `stock_id` at `price` (with slippage + commission)."""
        exec_price = price * (1.0 + self.slippage_pct)
        gross = shares * exec_price
        commission = gross * self.commission_pct
        total_cost = gross + commission

        self.cash -= total_cost
        self.positions[stock_id] = self.positions.get(stock_id, 0.0) + shares

        trade = Trade(
            date=date,
            stock_id=stock_id,
            action="BUY",
            shares=shares,
            price=price,
            exec_price=exec_price,
            gross_value=gross,
            commission=commission,
            net_cash_flow=-total_cost,
        )
        self._trade_log.append(vars(trade))
        return trade

    def _execute_sell(
        self,
        date: pd.Timestamp,
        stock_id: str,
        shares: float,
        price: float,
    ) -> Trade:
        """Sell `shares` of `stock_id` at `price` (with slippage + commission)."""
        exec_price = price * (1.0 - self.slippage_pct)
        gross = shares * exec_price
        commission = gross * self.commission_pct
        proceeds = gross - commission

        self.cash += proceeds
        held = self.positions.get(stock_id, 0.0)
        new_held = held - shares
        if abs(new_held) < 1e-6:
            self.positions.pop(stock_id, None)
        else:
            self.positions[stock_id] = max(new_held, 0.0)

        trade = Trade(
            date=date,
            stock_id=stock_id,
            action="SELL",
            shares=shares,
            price=price,
            exec_price=exec_price,
            gross_value=gross,
            commission=commission,
            net_cash_flow=proceeds,
        )
        self._trade_log.append(vars(trade))
        return trade

    # ─── Rebalancing ──────────────────────────────────────────────────────────

    def rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        date: pd.Timestamp,
    ) -> None:
        """
        Rebalance portfolio to match `target_weights`.

        Parameters
        ----------
        target_weights : dict
            {stock_id: weight} where weights sum to <= 1.0.
            Remainder stays in cash (already accounted for by caller).
        prices : dict
            {stock_id: current_price}  — raw prices (slippage applied internally)
        date : pd.Timestamp
            Rebalance date (for trade log entries).
        """
        nav = self.current_nav(prices)
        if nav <= 0:
            logger.warning(f"[{date.date()}] NAV is {nav:.2f}, skipping rebalance.")
            return

        # Compute target share counts
        target_shares: Dict[str, float] = {}
        for sid, weight in target_weights.items():
            p = prices.get(sid)
            if p and p > 0:
                target_shares[sid] = (nav * weight) / p
            else:
                target_shares[sid] = 0.0

        # All stocks we currently hold or want to hold
        all_stocks = set(self.positions.keys()) | set(target_shares.keys())

        # First pass: sell positions we no longer want (or are reducing)
        for sid in all_stocks:
            current = self.positions.get(sid, 0.0)
            target = target_shares.get(sid, 0.0)
            diff = target - current
            p = prices.get(sid, 0.0)
            if diff < -1e-6 and p > 0:
                # Need to sell |diff| shares
                self._execute_sell(date, sid, abs(diff), p)

        # Second pass: buy positions we want more of
        for sid in all_stocks:
            current = self.positions.get(sid, 0.0)
            target = target_shares.get(sid, 0.0)
            diff = target - current
            p = prices.get(sid, 0.0)
            if diff > 1e-6 and p > 0:
                # Check we have enough cash (after slippage + commission)
                cost_per_share = p * (1.0 + self.slippage_pct) * (1.0 + self.commission_pct)
                affordable = self.cash / cost_per_share
                actual_diff = min(diff, affordable)
                if actual_diff > 1e-6:
                    self._execute_buy(date, sid, actual_diff, p)

    # ─── Reporting ────────────────────────────────────────────────────────────

    def get_trade_log(self) -> List[dict]:
        """Return a copy of the trade log as a list of dicts."""
        return list(self._trade_log)
