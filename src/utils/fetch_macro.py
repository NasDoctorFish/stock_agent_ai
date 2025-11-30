import yfinance as yf
import pandas as pd
from fredapi import Fred
from datetime import datetime
from config.schema_macro import UNIFIED_MACRO_COLUMNS
# 병렬 I/O 처리 for loop보다 10배는 더 빠름
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ✅ 1️⃣ MultiIndex 컬럼 자동 flatten 함수
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]
    return df
import pandas as pd

# ✅ 시계열용 FREquency 생성 함수 FRED나 yfinance에는 없음.
def detect_frequency(df: pd.DataFrame, date_col: str = "date") -> str:
    """
    주어진 시계열 DataFrame에서 date 간격을 분석하여
    freq(D/M/Q/A) 값을 자동으로 추정합니다.
    """
    if date_col not in df.columns or len(df) < 2:
        return None

    # 날짜 차이 계산
    diffs = df[date_col].sort_values().diff().dropna().dt.days
    median_diff = diffs.median()

    # 대표적인 일수 기준
    if median_diff <= 2:
        return "D"  # Daily
    elif median_diff <= 31:
        return "M"  # Monthly
    elif median_diff <= 95:
        return "Q"  # Quarterly
    elif median_diff <= 370:
        return "A"  # Annual
    else:
        return None


def apply_unified_schema(df: pd.DataFrame, source: str, defaults: dict = None):
    """
    모든 macro 데이터프레임에 통일된 컬럼 스키마를 적용합니다.
    누락된 컬럼은 None 또는 defaults로 채웁니다.
    """
    if defaults is None:
        defaults = {}

    # 기본 필드
    df["source"] = source
    df["retrieved_at"] = datetime.now().strftime("%Y-%m-%d")

    # 모든 누락된 컬럼 채우기
    for col in UNIFIED_MACRO_COLUMNS:
        if col not in df.columns:
            df[col] = defaults.get(col, None)

    # 순서 통일
    df = df[UNIFIED_MACRO_COLUMNS]

    return df


# ✅ 2️⃣ 지표 데이터 수집 (확장성 높은 버전)
# src/utils/fetch_macro.py


def fetch_macro_indices(indices_dict, start_date, end_date, fred_api_key=None, logger=None, max_workers=8):
    """
    ✅ 병렬 FRED + yfinance 거시지표 수집 함수 (long-form)
    Returns: long-form DataFrame [date, index_name, value_norm, ...]
    """

    fred = Fred(api_key=fred_api_key) if fred_api_key else None
    indices = indices_dict.get("macro_indices_dict", indices_dict)

    success, failed = [], []

    # -------------------------------
    # 내부 fetch 함수 (각 인덱스 개별 처리)
    # -------------------------------
    def fetch_one(name, code):
        df = None
        try:
            # 1️⃣ FRED 시도
            if fred and code.isupper():
                fred_series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
                if fred_series is not None and not fred_series.empty:
                    df = fred_series.reset_index()
                    df.columns = ["date", "value_norm"]
                    df = apply_unified_schema(df, source="FRED")
                    df["ticker"] = code
                    df["freq"] = detect_frequency(df, "date")
                    df["index_name"] = name
                    if logger:
                        logger.info(f"📈 FRED: {name} ({code}) — {len(df)} points")
            # 2️⃣ yfinance 시도
            if df is None:
                data = yf.download(code, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if data.empty:
                    if logger:
                        logger.warning(f"⚠️ No yfinance data for {name} ({code})")
                    return None
                data = data[["Close"]].reset_index()
                data.columns = ["date", "value_norm"]
                data = apply_unified_schema(data, source="yfinance")
                data["ticker"] = code
                data["freq"] = detect_frequency(data, "date")
                data["index_name"] = name
                if logger:
                    logger.info(f"✅ yfinance: {name} ({code}) — {len(data)} points")
                df = data

            return df

        except Exception as e:
            if logger:
                logger.error(f"❌ Failed to fetch {name} ({code}): {e}")
            return None

    # -------------------------------
    # 병렬 실행
    # -------------------------------
    all_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, name, code): name for name, code in indices.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result is not None:
                    all_data.append(result)
                    success.append(name)
                else:
                    failed.append(name)
            except Exception as e:
                failed.append(name)
                if logger:
                    logger.error(f"❌ Thread failed for {name}: {e}")

    # -------------------------------
    # 데이터 병합 + 정렬
    # -------------------------------
    if not all_data:
        if logger:
            logger.error("❌ No macro data fetched at all.")
        return pd.DataFrame(columns=UNIFIED_MACRO_COLUMNS)

    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.sort_values(["index_name", "date"]).drop_duplicates()

    # -------------------------------
    # long-form 변환
    # -------------------------------
    long_df = (
        df_all
        .reindex(columns = UNIFIED_MACRO_COLUMNS)
        .reset_index(drop=True)
    )

    # -------------------------------
    # 메타 통계 출력
    # -------------------------------
    if logger:
        logger.info(f"📊 Macro indices fetched: {len(success)} succeeded, {len(failed)} failed.")
        if failed:
            logger.warn(f"⚠️ Failed indices: {failed}")
        logger.info(f"✅ Final long_df shape: {long_df.shape}")

    return long_df



# def fetch_macro_indices(indices_dict, start_date, end_date, fred_api_key=None, logger=None):
#     """
#     Fetch macroeconomic and financial indices from Yahoo Finance and FRED.
#     Compatible with both yfinance tickers (e.g. ^GSPC) and FRED codes (e.g. DGS10, M2SL).

#     Args:
#         indices_dict (dict): { "S&P500": "^GSPC", "DGS10": "DGS10", ... }
#         start_date (date or str)
#         end_date (date or str)
#         fred_api_key (str, optional)
#         logger (logging.Logger, optional): custom logger for structured output

#     Returns:
#         pd.DataFrame with columns: [date, index_name, value_norm]
#     """

#     fred = Fred(api_key=fred_api_key) if fred_api_key else None
#     indices = indices_dict.get("macro_indices_dict", indices_dict)


#     all_data = []
#     success, failed = [], []

#     for name, code in indices.items():
#         try:
#             df = None

#             # -------------------------------
#             # 1️⃣ FRED로 시도 (금리, M2, CPI 등)
#             # -------------------------------
#             if fred and code.isupper():
#                 try:
#                     fred_series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
#                     if fred_series is not None and not fred_series.empty:
#                         df = fred_series.reset_index()
#                         df.columns = ["date", "value_norm"]
                        
#                         # refactor the columns into UNIFIED_COLUMNS
#                         df = apply_unified_schema(df, source="FRED")
#                         df["ticker"] = code
#                         df["freq"] = detect_frequency(df = df, date_col = "date") 
# # df_fred = df_fred.reindex(columns=UNIFIED_COLUMNS)
# # df_yf = df_yf.reindex(columns=UNIFIED_COLUMNS)
# # df_all = pd.concat([df_yf, df_fred], ignore_index=True)
#                         df["index_name"] = name
#                         success.append(name)
#                         if logger:
#                             logger.info(f"📈 FRED: {name} ({code}) — {len(df)} points")
#                     else:
#                         if logger:
#                             logger.warn(f"⚠️ No FRED data for {name} ({code})")

#                 except Exception as e:
#                     if logger:
#                         logger.warn(f"⚠️ FRED fetch failed for {name} ({code}): {e}")

#             # -------------------------------
#             # 2️⃣ yfinance로 시도 (주가, 환율 등)
#             # -------------------------------
#             if df is None:
#                 data = yf.download(code, start=start_date, end=end_date, progress=False, auto_adjust=True)
#                 if data.empty:
#                     failed.append(name)
#                     if logger:
#                         logger.warn(f"⚠️ No yfinance data for {name} ({code})")
#                     continue
                
#                 data = data[["Close"]].reset_index()
#                 data.columns = ["date", "value_norm"] # change column names to these
#                 data = apply_unified_schema(df= data, source="yfinance")
#                 data["ticker"] = code
#                 data["freq"] = detect_frequency(df = data, date_col = "date") 
#                 data["index_name"] = name # we are iterating name now
#                 df = data
#                 success.append(name) # success log
#                 if logger:
#                     logger.info(f"✅ yfinance: {name} ({code}) — {len(df)} points")

#             all_data.append(df)

#         except Exception as e:
#             failed.append(name)
#             if logger:
#                 logger.error(f"❌ Failed to fetch {name} ({code}): {e}")

#     # -------------------------------
#     # 3️⃣ 병합 및 전처리
#     # -------------------------------
#     if not all_data:
#         if logger:
#             logger.error("❌ No macro data fetched at all.")
#         return pd.DataFrame(columns=UNIFIED_MACRO_COLUMNS)

#     df_all = pd.concat(all_data, ignore_index=True)
#     df_all = df_all.sort_values(["index_name", "date"]).drop_duplicates()

#     # Long-form으로 변환
#     long_df = df_all.reset_index().melt(id_vars="date", var_name="index_name", value_name="value_norm")

#     # -------------------------------
#     # 5️⃣ 메타 통계 출력
#     # -------------------------------
#     if logger:
#         logger.info(f"📊 Macro indices fetched: {len(success)} succeeded, {len(failed)} failed.")
#         if failed:
#             logger.warn(f"⚠️ Failed indices: {failed}")
#         logger.info(f"✅ Final long_df shape: {long_df.shape}")

#     return long_df


# def fetch_macro_indices_FRED(indices, start_date, end_date, FRED_API_KEY):
#     '''
#     Fetch multiple macroeconomic indicators from FRED.
    
#     datatype: JSON-like
#     example record:
#     {
#         "realtime_start": "2025-10-29",
#         "realtime_end": "2025-10-29",
#         "date": "1974-01-01",
#         "value": "6145.506"
#     }

#     Parameters:
#         indices (list): list of FRED Series IDs (e.g., ['CPIAUCSL', 'FEDFUNDS', 'SP500'])
#         start_date (str): start date in 'YYYY-MM-DD' format
#         end_date (str): end date in 'YYYY-MM-DD' format
#         FRED_API_KEY (str): your FRED API key
#     Returns:
#         pd.DataFrame: concatenated dataframe with columns ['date', 'value', 'index_name']
#     '''

#     fred = Fred(api_key=FRED_API_KEY)
#     df_list = []

#     for idx in indices:
#         try:
#             # fetch data from FRED
#             data = fred.get_series(idx, observation_start=start_date, observation_end=end_date)
#             df = data.reset_index()
#             df.columns = ["date", "value"]
#             df["index_name"] = idx
#             df_list.append(df)
#         except Exception as e:
#             print(f"⚠️ Failed to fetch {idx}: {e}")
#             continue

#     if not df_list:
#         return pd.DataFrame(columns=["date", "value", "index_name"])

#     return pd.concat(df_list, ignore_index=True)