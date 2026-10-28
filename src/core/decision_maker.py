"""
DecisionMaker
-------------
Integrates multiple analysts and produces actionable investment decisions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from datetime import datetime

import yaml


def _load_benchmark_tickers() -> set:
    """Tickers present in price data for NAV comparison only (see
    config.yaml::benchmark_tickers) — never eligible as BUY/SELL candidates."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return set(cfg.get("benchmark_tickers", []))
    except Exception:
        return set()


class DecisionMaker:
    def __init__(self, investor, logger=None):
        self.investor = investor
        self.logger = logger

    def run(self) -> Dict:
        context = self.investor.aggregate()
        decision = self.make_decision(context)
        if self.logger:
            self.logger.info(f"Final Decision: {decision}")
        print("\nFinal Investment Decision")
        print(decision)
        return decision

    def make_decision(self, context: Dict) -> Dict:
        stock_data = context.get("stock_data", {})
        fundamentals = context.get("fundamental", {})
        news = context.get("news", {})
        business = context.get("business", {})

        regime = stock_data.get("data", {}).get("economic_regime", "unknown")
        regime_metrics = stock_data.get("data", {}).get("regime_metrics", {})

        equities_favorable = regime in {"growth", "neutral"}

        scored = self._score_stocks(stock_data, fundamentals, news, business)
        top_stocks = sorted(scored, key=lambda x: x["score"], reverse=True)[:10]

        alternatives = []
        if not equities_favorable:
            alternatives = [
                "cash",
                "short-duration bonds",
                "investment-grade bonds",
                "gold/commodities",
                "defensive equity sectors",
            ]

        recommendation = (
            "Equities appear favorable based on the detected regime."
            if equities_favorable
            else "Equities look unfavorable; prioritize defensive alternatives."
        )

        return {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "economic_regime": regime,
            "regime_metrics": regime_metrics,
            "equities_favorable": equities_favorable,
            "top_stocks": top_stocks,
            "alternatives": alternatives,
            "recommendation": recommendation,
            "context_summary": {
                "macro": context.get("macro", {}).get("data", {}),
                "fundamental": fundamentals.get("summary", ""),
                "stock_data": stock_data.get("summary", ""),
                "news": news.get("summary", ""),
                "business": business.get("summary", ""),
            },
        }

    def _score_stocks(self, stock_data: Dict, fundamentals: Dict, news: Dict, business: Dict) -> List[Dict]:
        stock_metrics = stock_data.get("data", {}).get("stocks", {})
        fundamental_metrics = fundamentals.get("data", {}).get("stocks", {})
        news_metrics = news.get("data", {}).get("stocks", {})
        business_metrics = business.get("data", {}).get("stocks", {})

        stock_ids = set(stock_metrics) | set(fundamental_metrics) | set(news_metrics) | set(business_metrics)
        stock_ids -= _load_benchmark_tickers()
        results = []
        for stock_id in stock_ids:
            metrics = {}
            reasons = []
            score = 0.0

            s = stock_metrics.get(stock_id, {})
            f = fundamental_metrics.get(stock_id, {})
            n = news_metrics.get(stock_id, None)
            b = business_metrics.get(stock_id, {})

            if s.get("score") is None:
                continue  # needs >252 days of history for 12-1 momentum
            score = s["score"]
            if score > 0:
                reasons.append("positive 12-1 momentum")

            metrics.update(s)
            metrics.update(f)
            metrics["sentiment"] = float(n) if n is not None else None
            metrics.update(b)

            results.append({
                "stock_id": stock_id,
                "score": score,
                "reasons": reasons,
                "metrics": metrics,
            })

        return results
