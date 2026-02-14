from __future__ import annotations

from analysts.base_analyst import BaseAnalyst, AnalystResult
import pandas as pd


class NewsMediaAnalyst(BaseAnalyst):
    name = "news"

    def analyze(self) -> AnalystResult:
        if self.db is None:
            return self._result(False, errors=["No DB handler provided"])

        if not self.db.table_exists("news_sentiment"):
            return self._result(False, errors=["Table news_sentiment not found"])

        query = "SELECT stock_id, date, sentiment_score FROM news_sentiment"
        try:
            df = self.db.query(query)
        except Exception as e:
            return self._result(False, errors=[f"Query failed: {e}"])

        if df is None or df.empty:
            return self._result(False, summary="No news sentiment data")

        df = df.dropna(subset=["stock_id", "date", "sentiment_score"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # Focus on recent news (last 30 days)
        if not df.empty:
            cutoff = df["date"].max() - pd.Timedelta(days=30)
            df = df[df["date"] >= cutoff]

        stocks = (
            df.groupby("stock_id")["sentiment_score"]
            .mean()
            .sort_values(ascending=False)
            .to_dict()
        )

        top_stocks = [
            {"stock_id": k, "sentiment": float(v)}
            for k, v in list(stocks.items())[:10]
        ]

        return self._result(
            True,
            data={"stocks": stocks, "top_stocks": top_stocks},
            summary="News sentiment analysis completed",
        )
