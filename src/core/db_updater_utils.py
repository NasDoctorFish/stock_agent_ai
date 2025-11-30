# db_updater_utils.py
import pandas as pd
from datetime import datetime, timedelta
from utils.fetch_yfinance import fetch_prices
from utils.fetch_newsapi import fetch_news_sentiment
from utils.fetch_macro import fetch_macro_indices
from dotenv import load_dotenv
from pathlib import Path
import os
import yaml


# ============================
# ✅ 환경 설정 및 경로 로드
# ============================
def load_settings_and_keys(self):
    # ✅ base_dir은 항상 절대경로 기준으로
    BASE_DIR = Path(__file__).resolve().parents[2]   # "Stock ML" 폴더
    CONFIG_PATH = BASE_DIR / "src" / "config" / "config.yaml"

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    self.FRED_INDICES = config["fred_indices"]

    load_dotenv()
    self.NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    self.FRED_API_KEY = os.getenv("FRED_API_KEY")
    self.FMP_API_KEY = os.getenv("FMP_API_KEY")
    self.EXCHANGERATE_API_KEY = os.getenv("EXCHANGERATE_API_KEY")

    self.DATA_PATH = "data/raw"
    self.TODAY = datetime.now().date()
    self.START_DATE = self.TODAY - timedelta(days=365)  # 최근 1개월치만 업데이트

# ============================
# ✅ 함수 정의
# ============================

# def fetch_top_tickers(self, limit=1000):
#     url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={self.FMP_API_KEY}"
#     r = requests.get(url)
#     r.raise_for_status()
#     df = pd.DataFrame(r.json())

#     # marketCap 기준으로 정렬
#     df = df[df["marketCap"].notna()]
#     df = df.sort_values("marketCap", ascending=False).head(limit)

#     tickers = df["symbol"].dropna().unique().tolist()
#     print(f"✅ {len(tickers)} tickers loaded (Top {limit} by market cap)")
#     return tickers

def load_existing(self, path):
    """CSV 파일 로드 + date 컬럼 처리 (빈 파일 방어 포함)"""
    import os
    try:
        if not os.path.exists(path):
            self.logger.warning(f"⚠️ File not found: {path}, creating empty DataFrame.")
            return pd.DataFrame()
        if os.path.getsize(path) == 0:
            self.logger.warning(f"⚠️ File is empty: {path}")
            return pd.DataFrame()

        cols = pd.read_csv(path, nrows=0).columns
        date_col = "date" if "date" in cols else "fiscal_date" if "fiscal_date" in cols else None

        if date_col:
            df = pd.read_csv(path, parse_dates=[date_col]).rename(columns={date_col: "date"})
            self.logger.info(f"✅ Loaded existing file: {path} ({len(df)} rows)")
            return df
        else:
            self.logger.warning(f"⚠️ No date/fiscal_date column in {path}")
            return pd.read_csv(path)
    except Exception as e:
        self.logger.error(f"❌ Error loading file: {path} ({e})")
        return pd.DataFrame()



def merge_new(self, old, new, key_cols):
    """기존 데이터와 새 데이터를 병합"""
    if old is None or old.empty:
        self.logger.info("⚠️ Old dataset empty, returning new data only.")
        return new.copy()
    if new is None or new.empty:
        self.logger.info("⚠️ New dataset empty, keeping old data only.")
        return old.copy()

    df = pd.concat([old, new], ignore_index=True)
    df = df.drop_duplicates(subset=key_cols, keep="last")
    df = df.sort_values(key_cols)
    self.logger.info(f"✅ Merged data — now {len(df)} rows after deduplication.")
    return df


# =============================
# ✅ 메인 프로세스
# =============================

def update_csv_files(self):
    try:
        self.logger.info("===== Update process started =====")

        # ---- 단계별 실행 스위치 ----
        RUN_FETCH = True          # 가격/재무 데이터
        RUN_SENTIMENT = True      # 뉴스 감정 분석
        RUN_MACRO = True          # 거시지표
        RUN_ANALYSIS = True       # 분석/머징/저장

        # ---- 1️⃣ 기존 데이터 로드 ----
        price_old = self.load_existing(f"{self.DATA_PATH}/price_daily.csv")
        fund_old = self.load_existing(f"{self.DATA_PATH}/fundamentals_quarterly.csv")
        news_old = self.load_existing(f"{self.DATA_PATH}/news_sentiment.csv")
        macro_old = self.load_existing(f"{self.DATA_PATH}/macro_index.csv")

        # ---- Load tickers Top 1000 단계 ----
        # 노가다로 내가 원하는 주식만 뽑아오기 필요
        CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

        #config.yaml에서 ticker 뽑아오기
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)

        tickers = cfg["tickers"]
        self.logger.info(f"Loaded {len(tickers)} tickers")
        
        # ---- 2️⃣ Fetch 단계 ----
        if RUN_FETCH:
            self.logger.info("Fetching latest prices & fundamentals...")
            price_new, fund_new = fetch_prices(tickers, self.START_DATE, self.TODAY)
        else:
            price_new, fund_new = None, None

        if RUN_SENTIMENT:
            self.logger.info("Fetching news sentiment data...")
            news_new = fetch_news_sentiment(tickers, self.START_DATE, self.TODAY, self.NEWS_API_KEY)
        else:
            news_new = None

        if RUN_MACRO:
            self.logger.info("Fetching macro indices data...")
            BASE_DIR = Path(__file__).resolve().parents[2]   # Stock ML 폴더
            MACRO_PATH = BASE_DIR / "src" / "config" / "macro_indices_dict.yaml"  # 예시 위치
            with open(MACRO_PATH, "r") as f:
                macro_indices_dict = yaml.safe_load(f)

            macro_new = fetch_macro_indices(
                indices_dict=macro_indices_dict,
                start_date="2000-01-01",
                end_date=self.TODAY,
                fred_api_key=self.FRED_API_KEY,
                logger=self.logger
            )
            print("macro_new cols:", macro_new.columns)
            print(macro_new.head())
        else:
            macro_new = None

        # ---- 3️⃣ Merge 단계 ----
        if RUN_ANALYSIS:
            self.logger.info("Merging new data with old datasets...")

            price_all = merge_new(self, price_old, price_new, ["stock_id", "date"]) if price_new is not None else price_old
            fund_all = merge_new(self, fund_old, fund_new, ["stock_id", "fiscal_date"]) if fund_new is not None else fund_old
            news_all = merge_new(self, news_old, news_new, ["stock_id", "date", "headline"]) if news_new is not None else news_old
            macro_all = merge_new(self, macro_old, macro_new, ["index_name", "date"]) if macro_new is not None else macro_old

            # ---- 4️⃣ Save 단계 ----
            self.logger.info("Saving merged data to CSV files...")
            price_all.to_csv(f"{self.DATA_PATH}/price_daily.csv", index=False)
            fund_all.to_csv(f"{self.DATA_PATH}/fundamentals_quarterly.csv", index=False)
            news_all.to_csv(f"{self.DATA_PATH}/news_sentiment.csv", index=False)
            macro_all.to_csv(f"{self.DATA_PATH}/macro_index.csv", index=False)

        self.logger.info(f"✅ All datasets updated successfully for {self.TODAY}")
        self.logger.info("===== Update process finished =====\n")

    except Exception as e:
        self.logger.exception("❌ Unexpected error during update process")


