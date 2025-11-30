# analysts/macro_analyst.py
from analysts.base_analyst import BaseAnalyst
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- LSTM 기반 매크로 트렌드 분석기 ---
class MacroTrendLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# --- MacroAnalyst 클래스 ---
class MacroAnalyst(BaseAnalyst):
    """
    Advanced macroeconomic regime detector using yfinance-based DuckDB data.
    """

    def __init__(self, db_handler, logger=None):
        super().__init__(logger)
        self.db = db_handler
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = MacroTrendLSTM().to(self.device)
        self.feat_cols = None
        self.logger.info(f"🧠 MacroAnalyst initialized on {self.device}")

    # ===================== 데이터 로드 =====================
    def load_macro_data(self):
        query = """
        SELECT date, value_norm, index_name, ticker, source, freq, change_pct, value_norm
        FROM macro_index_full
        WHERE date > DATE '2015-01-01'
        ORDER BY date ASC;
        """
        df = self.db.query(query)
        if df.empty:
            self.logger.warning("⚠️ No macro data found.")
        return df

    # ===================== 전처리 =====================
    def preprocess(self, df: pd.DataFrame):
        # ✅ 만약 long-form 구조라면 pivot 적용
        if "index_name" in df.columns and "value_norm" in df.columns:
            df = df.pivot(index="date", columns="index_name", values="value_norm")

        # ✅ 숫자 컬럼만 남기기
        df = df.select_dtypes(include=[float, int])

        df = df.ffill().dropna()
        df_scaled = (df - df.mean()) / df.std()

        return df_scaled

    # ===================== 모델 학습 =====================
    def train_lstm(self, df):
        X, y = [], []
        self.logger.info("Columns in MACRO DATA: %s", df.columns.tolist())

        # ✅ 타깃 대체: 유동성(M2SL ↑) + 금리(FEDFUNDS ↓) 기반 macro sentiment proxy
        if "M2SL" not in df.columns or "FEDFUNDS" not in df.columns:
            self.logger.warn("⚠️ 'M2SL' 또는 'FEDFUNDS' 컬럼이 없어 학습 스킵.")
            self.feat_cols = [c for c in df.columns]
            return False

        # 유동성 증가율(6개월 변화) + 금리 변화(6개월 차분)
        liquidity_signal = df["M2SL"].pct_change(6).shift(-6).fillna(0)
        rate_signal = -df["FEDFUNDS"].diff(6).shift(-6).fillna(0)

        # 두 신호를 합산 → 경기 확장기면 1, 수축기면 0
        combined_signal = liquidity_signal + rate_signal
        y = (combined_signal > 0).astype(float).values  # bullish=1, bearish=0

        seq_len = 6  # 6개월 시퀀스
        self.feat_cols = [c for c in df.columns if c not in ["M2SL", "FEDFUNDS"]]  # 예측용 feature 목록
        features = df[self.feat_cols].values
        input_dim = features.shape[1]
        self.model = MacroTrendLSTM(input_dim=input_dim).to(self.device)

        # LSTM 입력 시퀀스 구성
        for i in range(len(features) - seq_len):
            X.append(features[i:i + seq_len])

        X = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        y = torch.tensor(y[seq_len:], dtype=torch.float32).unsqueeze(1).to(self.device)

        # 방어 코드: 데이터 부족 시 학습 스킵
        if len(X) == 0 or len(y) == 0:
            self.logger.warn("⚠️ LSTM 학습에 필요한 시퀀스 데이터가 부족합니다. 학습 스킵.")
            return False

        # Optimizer & Loss 정의
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()

        # 학습 루프
        self.model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            preds = self.model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

        self.logger.info(f"✅ LSTM trained (Final Loss: {loss.item():.4f})")

        # ✅ 항상 False 반환 (학습 완료 후 명시적 종료)
        return False





    # ===================== 트렌드 분석 =====================
    def analyze(self):
            
        df_raw = self.load_macro_data()
        if df_raw.empty:
            return {"macro_trend": "unknown"}

        try:
            df_feat = self.preprocess(df_raw)
            self.feat_cols = df_feat.columns.tolist()
            self.logger.info(f"✅ Feature columns set: {self.feat_cols}")
            print("df_feat.columns", df_feat.columns)

            # 자동 감지 or json 불러오기 후
            self.feat_cols = [c for c in self.feat_cols if c in df_feat.columns and c is not None]

            missing = [c for c in self.feat_cols if c not in df_feat.columns]
            if missing:
                self.logger.warn(f"⚠️ Dropping missing features: {missing}")
        
            
            self.train_lstm(df_feat) # 여기서 에러 종종 발생: 데이터 타입 불일치
            self.logger.info(f"Macro Analyst: LSTM model train completed, features: {str(df_feat.columns)}")
        except ValueError as e:
            self.logger.error(f"Macro Analyze ValueError: df_feat might not be in correct format. Check the CSV file and new fetched data")
            self.logger.error(f"df_feat() Check: \n {str(df_feat.head())}")
            self.logger.error(f"df_feat() Column Check: \n {list(df_feat.columns)}")
        except Exception as e:
            self.logger.error(f"Macro Analyze Failed (Non-ValueError): {e}")
            
        # ✅ 여기! feat_cols 체크 추가
        if not self.feat_cols or len(self.feat_cols) == 0:
            self.logger.warn("⚠️ feat_cols is empty or None, skipping analysis.")
            return {"macro_trend": "unknown"}
        
        # 최근 데이터로 예측
        latest_seq = torch.tensor(
            df_feat[self.feat_cols].tail(12).values[None, :, :],
            dtype=torch.float32
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            prob = self.model(latest_seq).item()

        macro_trend = "bullish" if prob > 0.6 else "bearish" if prob < 0.4 else "neutral"

        latest = df_feat.tail(1)
        result = {
            "macro_trend": macro_trend,
            "prob_bullish": round(prob, 3),
            "latest_date": latest.index[-1].strftime("%Y-%m-%d"),
            # "inflation": round(latest["CPIAUCSL"].values[0], 3),
            # "m2": round(latest["M2SL"].values[0], 3),
            # "fed_rate": round(latest["FEDFUNDS"].values[0], 3)
        }

        self.logger.info(f"📊 Macro trend result: {result}")
        return result
    
    
    # 모든 데이터 float32로 강제 변환
    def _clean_numeric_df(df: pd.DataFrame, feature_list: list):
        """모든 feature를 float32로 강제 변환"""
        df = df.copy()
        for col in feature_list:
            # 문자열 섞인 숫자 처리 (%, 콤마, $, 공백)
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.replace('%', '', regex=False)
                .str.replace('$', '', regex=False)
                .str.strip()
            )
            # 숫자 변환 (실패 시 NaN)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 무한대 → NaN, NaN 보간
        df[feature_list] = df[feature_list].replace([np.inf, -np.inf], np.nan)
        df[feature_list] = df[feature_list].interpolate(limit_direction="both").ffill().bfill()
        return df

    
    
    # Streamlit + PyTorch 그래프 시각화
    # --- 메인 함수: 시각화 및 선택 기능 ---
    def visualize_macro_trend(self, df: pd.DataFrame):
        """
        LSTM 매크로 트렌드 예측 확률과 선택한 매크로 지표를 시각화합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            'date' 컬럼과 여러 macro feature를 포함한 데이터프레임
        model_path : str
            사전 학습된 LSTM 모델의 경로 (.pth)
        """

        # UI 제목
        st.title("📊 MacroTrend LSTM 시각화 도구")

        # 피처 선택
        feature_list = [c for c in df.columns if c != "date"]
        df = self._clean_numeric_df(df, feature_list)
        selected_feature = st.selectbox("비교할 매크로 지표를 선택하세요:", feature_list)

        # 모델 로드
        model = self.model
        model.eval()

        # 데이터 변환 및 예측
        X = torch.tensor(df[feature_list].values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = model(X).squeeze().numpy()

        df["prob_bullish"] = prob

        # 그래프 출력
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df["date"], df[selected_feature], color='tab:blue', label=selected_feature)
        ax1.set_ylabel(selected_feature, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(df["date"], df["prob_bullish"], color='tab:red', label="Prob. Bullish (Model)")
        ax2.set_ylabel("Prob. Bullish", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title(f"{selected_feature} vs LSTM 상승확률")
        plt.tight_layout()
        st.pyplot(fig)

