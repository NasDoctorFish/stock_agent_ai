"""
KR Decision Maker
=================
Korean-market-specific decision maker.  Reads the "kr_stock_data" key
produced by KRStockDataAnalyst instead of the "stock_data" key used by
the US pipeline's StockDataAnalyst.

Root bug this fixes:
    DecisionMaker.make_decision() does context.get("stock_data", {})
    but KRStockDataAnalyst.name = "kr_stock_data", so the key never
    matched and the KR pipeline always produced regime="unknown" with
    an empty top_stocks list.

Output format is identical to the US DecisionMaker so that the reporter,
scheduler, and order executor can treat both pipelines uniformly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class KRDecisionMaker:
    """
    Aggregates KRStockDataAnalyst output into an actionable KR investment decision.

    Args:
        investor: Investor instance whose analysts include KRStockDataAnalyst
        logger_:  StatusLogger or standard logger (optional)
    """

    # Regimes where Korean equities are considered tradeable
    FAVORABLE_REGIMES = {"growth", "neutral"}

    def __init__(self, investor, logger_=None):
        self.investor = investor
        self.logger = logger_ or logging.getLogger(__name__)

    def run(self) -> Dict[str, Any]:
        """Run all analysts via investor and return the KR decision dict."""
        context = self.investor.aggregate()
        decision = self.make_decision(context)
        self.logger.info(f"[KR Decision] regime={decision['economic_regime']} "
                         f"top={[s['stock_id'] for s in decision['top_stocks'][:3]]}")
        self._print(decision)
        return decision

    def make_decision(self, context: Dict) -> Dict[str, Any]:
        """
        Build the KR investment decision from the aggregated analyst context.

        Reads: "kr_stock_data" (from KRStockDataAnalyst)
        """
        kr_data = context.get("kr_stock_data", {})

        if not kr_data.get("success"):
            errors = kr_data.get("errors", ["KRStockDataAnalyst did not succeed"])
            self.logger.error(f"[KR Decision] KRStockDataAnalyst failed: {errors}")
            return self._empty_decision(errors=errors)

        data = kr_data.get("data", {})
        regime = data.get("economic_regime", "unknown")
        regime_metrics = data.get("regime_metrics", {})
        stocks = data.get("stocks", {})
        kiwoom_map = data.get("kiwoom_map", {})

        equities_favorable = regime in self.FAVORABLE_REGIMES

        # Score and rank stocks
        scored = self._score_stocks(stocks, kiwoom_map)
        top_stocks = sorted(scored, key=lambda x: x["score"], reverse=True)[:10]

        alternatives: List[str] = []
        if not equities_favorable:
            alternatives = [
                "cash (KRW)",
                "Korean short-term bonds (통안증권)",
                "gold ETF (KRX: 132030)",
                "defensive KR sectors (utilities, consumer staples)",
            ]

        recommendation = (
            "Korean equities appear favorable — consider top-ranked KR stocks."
            if equities_favorable
            else "Korean market in unfavorable regime — consider defensive alternatives."
        )

        return {
            "market": "KR",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "economic_regime": regime,
            "regime_metrics": regime_metrics,
            "equities_favorable": equities_favorable,
            "top_stocks": top_stocks,
            "alternatives": alternatives,
            "recommendation": recommendation,
            "context_summary": {
                "kr_stock_data": kr_data.get("summary", ""),
            },
        }

    def _score_stocks(
        self, stocks: Dict[str, Dict], kiwoom_map: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Score each stock by its technical metrics.

        Score = 2.0 * momentum  - 0.5 * volatility + 0.1 * drawdown
        (Same weighting as US StockDataAnalyst for consistent comparison.)
        """
        results = []
        for stock_id, metrics in stocks.items():
            momentum = metrics.get("momentum") or 0.0
            volatility = metrics.get("volatility") or 0.0
            drawdown = metrics.get("drawdown") or 0.0

            score = 2.0 * momentum - 0.5 * volatility + 0.1 * drawdown
            reasons: List[str] = []
            if momentum > 0:
                reasons.append("positive momentum")
            if volatility < 0.01:
                reasons.append("low volatility")
            if drawdown > -0.05:
                reasons.append("shallow drawdown")

            results.append({
                "stock_id": stock_id,
                "kiwoom_code": kiwoom_map.get(stock_id, ""),
                "score": round(score, 6),
                "reasons": reasons,
                "metrics": metrics,
            })
        return results

    def _empty_decision(self, errors: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "market": "KR",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "economic_regime": "unknown",
            "regime_metrics": {},
            "equities_favorable": False,
            "top_stocks": [],
            "alternatives": [],
            "recommendation": "KR analysis failed — no actionable signal.",
            "errors": errors or [],
            "context_summary": {},
        }

    def _print(self, decision: Dict) -> None:
        """Print a concise KR decision summary to stdout."""
        regime = decision["economic_regime"]
        favorable = decision["equities_favorable"]
        top = decision["top_stocks"][:5]
        print(f"\n{'='*50}")
        print(f"  [KR] Market Decision")
        print(f"  Regime     : {regime}")
        print(f"  Favorable  : {favorable}")
        print(f"  Top stocks :")
        for s in top:
            kcode = s.get("kiwoom_code", "")
            print(f"    {s['stock_id']:20s} ({kcode})  score={s['score']:+.4f}  {', '.join(s['reasons'])}")
        if not top:
            print("    (no stocks scored)")
        print(f"  Note: {decision['recommendation']}")
        print(f"{'='*50}\n")
