from __future__ import annotations

from analysts.base_analyst import BaseAnalyst, AnalystResult
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - fallback when torch isn't available
    torch = None
    nn = None


# --- LSTM 기반 매크로 트렌드 분석기 ---
class MacroTrendLSTM(nn.Module if nn is not None else object):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, output_dim=1):
        if nn is None:
            raise RuntimeError("Torch is not available")
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


class MacroAnalyst(BaseAnalyst):
    """
    Advanced macroeconomic regime detector using DuckDB data.
    """

    name = "macro"

    def __init__(self, db_handler, logger=None):
        super().__init__(db_handler, logger)
        self.device = None
        self.model = None
        self.model_trained = False
        self.feat_cols = None

        if torch is not None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.model = MacroTrendLSTM().to(self.device)
            self.logger.info(f"🧠 MacroAnalyst initialized on {self.device}")
        else:
            self.logger.warn("⚠️ Torch not available. MacroAnalyst will use heuristic mode.")

    def load_macro_data(self) -> pd.DataFrame:
        query = """
        SELECT date, value_norm, index_name, ticker, source, freq, change_pct, value_norm
        FROM macro_index_full
        WHERE date > DATE '2015-01-01'
        ORDER BY date ASC;
        """
        try:
            df = self.db.query(query)
        except Exception as e:
            self.logger.error(f"MacroAnalyst: failed to query macro data: {e}")
            return pd.DataFrame()
        if df is None or df.empty:
            self.logger.warn("⚠️ No macro data found.")
            return pd.DataFrame()
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if "index_name" in df.columns and "value_norm" in df.columns:
            df = df.pivot(index="date", columns="index_name", values="value_norm")
        df = df.select_dtypes(include=[float, int])
        df = df.ffill().dropna()
        if df.empty:
            return df
        df_scaled = (df - df.mean()) / df.std()
        return df_scaled

    def train_lstm(self, df: pd.DataFrame) -> bool:
        if torch is None or self.model is None:
            return False

        if not self.feat_cols:
            self.feat_cols = [c for c in df.columns]

        if not self.feat_cols:
            self.logger.warn("⚠️ No features available for LSTM training.")
            return False

        if "M2SL" not in df.columns or "FEDFUNDS" not in df.columns:
            self.logger.warn("⚠️ 'M2SL' 또는 'FEDFUNDS' 컬럼이 없어 학습 스킵.")
            return False

        liquidity_signal = df["M2SL"].pct_change(6).shift(-6).fillna(0)
        rate_signal = -df["FEDFUNDS"].diff(6).shift(-6).fillna(0)
        combined_signal = liquidity_signal + rate_signal
        y = (combined_signal > 0).astype(float).values

        seq_len = 6
        features = df[self.feat_cols].values
        input_dim = features.shape[1]
        self.model = MacroTrendLSTM(input_dim=input_dim).to(self.device)

        X = []
        for i in range(len(features) - seq_len):
            X.append(features[i:i + seq_len])

        if not X:
            self.logger.warn("⚠️ LSTM 학습에 필요한 시퀀스 데이터가 부족합니다. 학습 스킵.")
            return False

        X = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        y = torch.tensor(y[seq_len:], dtype=torch.float32).unsqueeze(1).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()

        self.model.train()
        for _ in range(25):
            optimizer.zero_grad()
            preds = self.model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

        self.logger.info(f"✅ LSTM trained (Final Loss: {loss.item():.4f})")
        self.model_trained = True
        return True

    def _heuristic_prob(self, df_feat: pd.DataFrame) -> float:
        composite = df_feat.mean(axis=1)
        if composite.empty:
            return 0.5
        z = (composite.iloc[-1] - composite.mean()) / (composite.std() or 1.0)
        prob = 1 / (1 + np.exp(-z))
        return float(prob)

    def analyze(self) -> AnalystResult:
        df_raw = self.load_macro_data()
        if df_raw.empty:
            return self._result(False, data={"macro_trend": "unknown"}, summary="No macro data")

        try:
            df_feat = self.preprocess(df_raw)
        except Exception as e:
            self.logger.error(f"Macro Analyze Failed: {e}")
            return self._result(False, data={"macro_trend": "unknown"}, errors=[str(e)])

        if df_feat.empty:
            return self._result(False, data={"macro_trend": "unknown"}, summary="Empty macro features")

        if not self.feat_cols:
            self.feat_cols = list(df_feat.columns)

        trained = self.train_lstm(df_feat)
        prob = None
        model_used = "heuristic"

        if trained and torch is not None and self.model is not None:
            try:
                latest_seq = torch.tensor(
                    df_feat[self.feat_cols].tail(12).values[None, :, :],
                    dtype=torch.float32,
                ).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    prob = float(self.model(latest_seq).item())
                model_used = "lstm"
            except Exception as e:
                self.logger.warn(f"⚠️ LSTM inference failed; fallback to heuristic. {e}")

        if prob is None:
            prob = self._heuristic_prob(df_feat)

        macro_trend = "bullish" if prob > 0.6 else "bearish" if prob < 0.4 else "neutral"
        latest_date = df_feat.index[-1]
        if hasattr(latest_date, "strftime"):
            latest_date = latest_date.strftime("%Y-%m-%d")

        result = {
            "macro_trend": macro_trend,
            "prob_bullish": round(prob, 3),
            "latest_date": latest_date,
            "model_used": model_used,
        }

        self.logger.info(f"📊 Macro trend result: {result}")
        return self._result(
            True,
            data=result,
            artifacts={"df_feat": df_feat},
            summary="Macro trend analysis completed",
        )

    def _clean_numeric_df(self, df: pd.DataFrame, feature_list: list):
        df = df.copy()
        for col in feature_list:
            if col not in df.columns:
                self.logger.warn(f"⚠️ Column {col} not in df, skipping.")
                continue
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df[feature_list] = df[feature_list].replace([np.inf, -np.inf], np.nan)
        df[feature_list] = df[feature_list].interpolate(limit_direction="both").ffill().bfill()
        return df

    def visualize_macro_trend(self, df: pd.DataFrame, selected_feature="BRENT", returnfig: bool = False):
        print("📊 MacroTrend 시각화 도구")

        if df is None or df.empty:
            self.logger.warn("⚠️ No data provided for visualization.")
            return None

        if selected_feature not in df.columns:
            self.logger.warn(f"⚠️ Selected feature '{selected_feature}' not in df.")
            return None

        feature_list = self.feat_cols or []
        feature_list = [c for c in feature_list if c in df.columns]

        if not feature_list:
            self.logger.warn("⚠️ No valid features available for visualization.")
            return None

        df = df.copy().reset_index()
        df = self._clean_numeric_df(df, feature_list)

        # 모델 또는 휴리스틱을 사용한 확률 계산
        if self.model_trained and torch is not None and self.model is not None:
            try:
                X = torch.tensor(df[feature_list].values, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prob = self.model(X).squeeze().cpu().numpy()
                df["prob_bullish"] = prob
            except Exception as e:
                self.logger.warn(f"LSTM visualize failed, fallback to heuristic: {e}")
                df["prob_bullish"] = 1 / (1 + np.exp(-df[feature_list].mean(axis=1)))
        else:
            df["prob_bullish"] = 1 / (1 + np.exp(-df[feature_list].mean(axis=1)))

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df["date"], df[selected_feature], color="tab:blue", label=selected_feature)
        ax1.set_ylabel(selected_feature, color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(df["date"], df["prob_bullish"], color="tab:red", label="Prob. Bullish")
        ax2.set_ylabel("Prob. Bullish", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        plt.title(f"{selected_feature} vs Macro Bullish Probability")
        plt.tight_layout()

        if returnfig:
            return fig
        plt.show()
        return None
