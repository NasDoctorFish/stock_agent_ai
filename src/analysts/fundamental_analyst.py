from __future__ import annotations

from analysts.base_analyst import BaseAnalyst, AnalystResult
import pandas as pd
import numpy as np


class FundamentalAnalyst(BaseAnalyst):
    name = "fundamental"

    def analyze(self) -> AnalystResult:
        if self.db is None:
            return self._result(False, errors=["No DB handler provided"])

        if not self.db.table_exists("fundamentals_q_full"):
            return self._result(False, errors=["Table fundamentals_q_full not found"])

        query = (
            "SELECT stock_id, fiscal_date, \"Total Revenue\" AS total_revenue, "
            "\"Net Income\" AS net_income "
            "FROM fundamentals_q_full"
        )
        try:
            df = self.db.query(query)
        except Exception as e:
            return self._result(False, errors=[f"Query failed: {e}"])

        if df is None or df.empty:
            return self._result(False, summary="No fundamentals data")

        df = df.dropna(subset=["stock_id", "fiscal_date"])
        df["fiscal_date"] = pd.to_datetime(df["fiscal_date"], errors="coerce")
        df = df.dropna(subset=["fiscal_date"])
        df = df.sort_values(["stock_id", "fiscal_date"])

        stocks = {}
        for stock_id, g in df.groupby("stock_id"):
            g = g.dropna(subset=["total_revenue", "net_income"], how="all")
            if len(g) < 1:
                continue
            latest = g.iloc[-1]
            prev = g.iloc[-2] if len(g) > 1 else None

            revenue_growth = None
            if prev is not None and pd.notna(prev["total_revenue"]) and prev["total_revenue"] != 0:
                revenue_growth = (latest["total_revenue"] - prev["total_revenue"]) / abs(prev["total_revenue"])

            net_margin = None
            if pd.notna(latest["total_revenue"]) and latest["total_revenue"] != 0:
                net_margin = latest["net_income"] / latest["total_revenue"] if pd.notna(latest["net_income"]) else None

            stocks[stock_id] = {
                "latest_fiscal_date": latest["fiscal_date"].strftime("%Y-%m-%d"),
                "revenue": float(latest["total_revenue"]) if pd.notna(latest["total_revenue"]) else None,
                "net_income": float(latest["net_income"]) if pd.notna(latest["net_income"]) else None,
                "revenue_growth": float(revenue_growth) if revenue_growth is not None else None,
                "net_margin": float(net_margin) if net_margin is not None else None,
            }

        if not stocks:
            return self._result(False, summary="No usable fundamentals")

        # Simple scoring: favor positive growth and margin
        scored = []
        for stock_id, metrics in stocks.items():
            score = 0.0
            if metrics["revenue_growth"] is not None:
                score += metrics["revenue_growth"]
            if metrics["net_margin"] is not None:
                score += metrics["net_margin"]
            scored.append({"stock_id": stock_id, "score": score, **metrics})

        top_stocks = sorted(scored, key=lambda x: x["score"], reverse=True)[:10]

        return self._result(
            True,
            data={"stocks": stocks, "top_stocks": top_stocks},
            summary="Fundamental analysis completed",
        )
