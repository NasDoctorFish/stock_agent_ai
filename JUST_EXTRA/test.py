# ==============================================================
# 🧠 비트코인(LSTM) 예측 통합 스크립트 (최종 안정 버전)
# ==============================================================

import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import logging
from datetime import datetime

# ==============================================================
# ✅ 경고 및 로그 설정
# ==============================================================

warnings.filterwarnings("ignore", message=".*no timezone found.*")

LOG_FILE = "btc_train_log.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ==============================================================
# ✅ 유저 입력 및 데이터 다운로드
# ==============================================================

print("비트코인: \"BTC-USD\"\n애플: 'AAPL'\n테슬라: 'TSLA'")

ticker = str(input("Ticker: ")).strip()
ticker = ticker.replace("_", "-").upper()  # ⚡ 자동 변환 (‘_’ → ‘-’)
start_date = "2022-01-01"
end_date = str(datetime.now().date())

try:
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")
    logging.info(f"✅ Data downloaded successfully for {ticker}")
except Exception as e:
    if "not found" in str(e).lower() or "possibly delisted" in str(e).lower():
        logging.error(f"⚠️ '{ticker}' 데이터를 찾을 수 없습니다. 예: BTC-USD 처럼 '-'을 사용하세요.")
    else:
        logging.exception(f"❌ Error downloading data for '{ticker}': {e}")
    exit()

# 저장
csv_name = f"{ticker}_price.csv"
data.to_csv(csv_name)
logging.info(f"📁 Saved raw data to {csv_name}")

# ==============================================================
# ✅ CSV 헤더 자동 감지 및 정리
# ==============================================================

first_two = pd.read_csv(csv_name, nrows=2)
is_multi = first_two.columns[0] != "Date" and "Ticker" in first_two.iloc[0].to_string()

if is_multi:
    logging.info("🧩 Detected multi-level header → fixing it...")
    df = pd.read_csv(csv_name, header=[0, 1], index_col=0)
    # 강제 표준 컬럼명 지정
    df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"][:df.shape[1]]
else:
    logging.info("✅ Single header detected")
    df = pd.read_csv(csv_name, header=0, index_col=0)

# 날짜 변환
df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
df.index.name = "Date"
df.to_csv(csv_name)
logging.info(f"✅ Cleaned CSV and saved to {csv_name}")
logging.info(f"Columns after cleaning: {list(df.columns)}")

# ==============================================================
# ✅ 데이터 전처리
# ==============================================================

data = df[['Close']]  # 반드시 DataFrame 형태 (Series ❌)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
X, y = create_sequences(scaled_data, SEQ_LENGTH)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

logging.info(f"✅ Data ready: X_train={X_train.shape}, X_test={X_test.shape}")

# ==============================================================
# ✅ 모델 구성 및 학습
# ==============================================================

def build_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info("🚀 Training started...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    logging.info("✅ Training complete.")
    return model

# ==============================================================
# ✅ 모델 파일 관리
# ==============================================================

model_name = f"{ticker}_lstm_model.keras"
file_path = os.path.join(os.getcwd(), model_name)

if os.path.exists(file_path):
    logging.info(f"✅ Found existing model: {file_path}")
    while True:
        userInput = input("실행(y), 업데이트(n): ").lower().strip()
        if userInput == 'y':
            model = load_model(file_path)
            logging.info("📦 Loaded existing model.")
            break
        elif userInput == 'n':
            model = build_model()
            model.save(file_path)
            logging.info(f"🔄 Model updated and saved as {file_path}")
            break
        else:
            print("⚠️ y 또는 n만 입력하세요.")
else:
    logging.info("❌ No model found — training new model...")
    model = build_model()
    model.save(file_path)
    logging.info(f"💾 Model saved as {file_path}")

# ==============================================================
# ✅ 예측 및 시각화
# ==============================================================

predictions = model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# y_test_rescaled와 predictions_rescaled의 길이만큼 날짜 인덱스 가져오기
test_dates = df.index[-len(y_test_rescaled):]

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_rescaled, label="Real Price")
plt.plot(test_dates, predictions_rescaled, label="Predicted Price")
plt.title(f"{ticker} Price Prediction (LSTM)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()


# ==============================================================
# ✅ 성능 평가
# ==============================================================

mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)

logging.info(f"MSE: {mse:.2f}")
logging.info(f"MAE: {mae:.2f}")

print("\n📊 Performance Summary:")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")

logging.info("🏁 Script finished successfully.")
