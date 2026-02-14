from __future__ import annotations

from analysts.base_analyst import BaseAnalyst, AnalystResult
import pandas as pd


class BusinessAnalyst(BaseAnalyst):
    name = "business"

    def analyze(self) -> AnalystResult:
        if self.db is None:
            return self._result(False, errors=["No DB handler provided"])

        if not self.db.table_exists("fundamentals_q_full"):
            return self._result(False, errors=["Table fundamentals_q_full not found"])

        query = (
            "SELECT stock_id, fiscal_date, \"Operating Income\" AS operating_income, "
            "\"Total Revenue\" AS total_revenue "
            "FROM fundamentals_q_full"
        )
        try:
            df = self.db.query(query)
        except Exception as e:
            return self._result(False, errors=[f"Query failed: {e}"])

        if df is None or df.empty:
            return self._result(False, summary="No business data")

        df = df.dropna(subset=["stock_id", "fiscal_date"])
        df["fiscal_date"] = pd.to_datetime(df["fiscal_date"], errors="coerce")
        df = df.dropna(subset=["fiscal_date"])
        df = df.sort_values(["stock_id", "fiscal_date"])

        stocks = {}
        for stock_id, g in df.groupby("stock_id"):
            g = g.dropna(subset=["operating_income", "total_revenue"], how="all")
            if len(g) < 1:
                continue
            latest = g.iloc[-1]
            op_margin = None
            if pd.notna(latest["total_revenue"]) and latest["total_revenue"] != 0:
                op_margin = latest["operating_income"] / latest["total_revenue"] if pd.notna(latest["operating_income"]) else None

            outlook = "neutral"
            if op_margin is not None:
                if op_margin > 0.15:
                    outlook = "positive"
                elif op_margin < 0.05:
                    outlook = "negative"

            stocks[stock_id] = {
                "latest_fiscal_date": latest["fiscal_date"].strftime("%Y-%m-%d"),
                "operating_margin": float(op_margin) if op_margin is not None else None,
                "outlook": outlook,
            }

        if not stocks:
            return self._result(False, summary="No usable business metrics")

        return self._result(
            True,
            data={"stocks": stocks},
            summary="Business analysis completed",
        )
