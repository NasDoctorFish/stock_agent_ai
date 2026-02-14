from __future__ import annotations

from analysts.base_analyst import BaseAnalyst, AnalystResult
import pandas as pd
import numpy as np


class StockDataAnalyst(BaseAnalyst):
    name = "stock_data"

    def analyze(self) -> AnalystResult:
        if self.db is None:
            return self._result(False, errors=["No DB handler provided"])

        if not self.db.table_exists("price_daily_full"):
            return self._result(False, errors=["Table price_daily_full not found"])

        query = "SELECT date, close, stock_id FROM price_daily_full WHERE close IS NOT NULL"
        try:
            df = self.db.query(query)
        except Exception as e:
            return self._result(False, errors=[f"Query failed: {e}"])

        if df is None or df.empty:
            return self._result(False, summary="No price data")

        df = df.dropna(subset=["date", "close", "stock_id"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values(["stock_id", "date"])

        df["log_return"] = df.groupby("stock_id")["close"].transform(lambda s: np.log(s).diff())

        window = 60
        min_obs = 20

        stocks = {}
        for stock_id, g in df.groupby("stock_id"):
            lr = g["log_return"].dropna()
            if len(lr) < min_obs:
                continue
            recent = lr.tail(window)
            cum = recent.cumsum()
            momentum = float(np.exp(cum.iloc[-1]) - 1) if len(cum) > 0 else None
            volatility = float(recent.std()) if len(recent) > 1 else None
            drawdown = float((cum - cum.cummax()).min()) if len(cum) > 1 else None
            stocks[stock_id] = {
                "momentum": momentum,
                "volatility": volatility,
                "drawdown": drawdown,
                "obs": int(len(recent)),
            }

        market_lr = df.groupby("date")["log_return"].mean().dropna()
        economic_regime = "unknown"
        regime_metrics = {}
        if len(market_lr) >= min_obs:
            recent = market_lr.tail(window)
            mean_lr = float(recent.mean())
            vol = float(recent.std()) if len(recent) > 1 else 0.0
            long_vol = float(market_lr.std()) if len(market_lr) > 1 else vol
            vol_cluster = float(recent.abs().autocorr(lag=1)) if len(recent) > 2 else 0.0

            if mean_lr > 0 and vol <= 1.2 * (long_vol or vol):
                economic_regime = "growth"
            elif mean_lr < 0:
                economic_regime = "drawdown"
            elif vol_cluster > 0.3 and vol > 1.2 * (long_vol or vol):
                economic_regime = "volatility_clustering"
            else:
                economic_regime = "neutral"

            regime_metrics = {
                "mean_log_return": mean_lr,
                "volatility": vol,
                "long_run_volatility": long_vol,
                "vol_cluster": vol_cluster,
            }

        scored = []
        for stock_id, metrics in stocks.items():
            momentum = metrics["momentum"] or 0.0
            vol = metrics["volatility"] or 0.0
            drawdown = metrics["drawdown"] or 0.0
            score = momentum - 0.5 * vol + 0.1 * drawdown
            scored.append({"stock_id": stock_id, "score": score, **metrics})

        top_stocks = sorted(scored, key=lambda x: x["score"], reverse=True)[:10]

        return self._result(
            True,
            data={
                "economic_regime": economic_regime,
                "regime_metrics": regime_metrics,
                "stocks": stocks,
                "top_stocks": top_stocks,
            },
            summary="Stock data analysis completed",
        )
