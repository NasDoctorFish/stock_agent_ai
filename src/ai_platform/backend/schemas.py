"""Pydantic models shared by the API, the AI tool schema and the job store."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Rebalance = Literal["weekly", "biweekly", "monthly", "quarterly", "custom"]


class EquityByRegime(BaseModel):
    risk_on: float = Field(1.0, ge=0, le=1)
    cautious: float = Field(0.7, ge=0, le=1)
    defensive: float = Field(0.4, ge=0, le=1)
    risk_off: float = Field(0.0, ge=0, le=1)


class SellRules(BaseModel):
    stop_loss_pct: Optional[float] = Field(None, gt=0, le=100, description="sell if down this % vs entry")
    take_profit_pct: Optional[float] = Field(None, gt=0, le=1000)
    max_holding_days: Optional[int] = Field(None, ge=1, le=3650)
    planned_sell_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD, liquidate all")


class Trigger(BaseModel):
    condition: str = Field(max_length=300)
    action: str = Field(max_length=300)


class Strategy(BaseModel):
    id: str = Field("", max_length=80)
    name: str = Field(max_length=120)
    rationale: str = Field("", max_length=3000)
    top_n: int = Field(10, ge=1, le=100)
    rebalance: Rebalance = "monthly"
    rebalance_days: Optional[int] = Field(None, ge=1, le=365)
    equity_by_regime: EquityByRegime = Field(default_factory=EquityByRegime)
    sell_rules: SellRules = Field(default_factory=SellRules)
    # descriptive only: the engine does not execute triggers yet
    rebalance_triggers: List[Trigger] = Field(default_factory=list, max_length=10)
    risk_notes: str = Field("", max_length=2000)


class BacktestParams(BaseModel):
    start: str
    end: str
    rebalance: Rebalance = "monthly"
    rebalance_days: Optional[int] = Field(None, ge=1, le=365)
    top_n: int = Field(10, ge=1, le=100)
    capital: float = Field(100_000.0, gt=0)
    currency: str = "USD"
    fx: float = Field(1.0, gt=0)           # units of currency per 1 USD
    benchmark: str = "SPY"
    slippage_pct: float = Field(0.1, ge=0)       # PERCENT, e.g. 0.1 = 0.1%
    commission_pct: float = Field(0.015, ge=0)   # PERCENT
    strategy: Optional[Strategy] = None


class AnalyzeRequest(BaseModel):
    model: Optional[str] = None
    mode: Literal["recommend", "full_ai"] = "recommend"
    params: BacktestParams
    notes: str = Field("", max_length=2000)
