# src/core/schema_macro.py
UNIFIED_MACRO_COLUMNS = [
    "date",             # 날짜
    "index_name",       # 예: 'CPIAUCSL', 'KRW/USD', 'S&P500'
    "ticker",           # 예: '^GSPC', 'KRW=X', FRED 시리즈 ID
    "source",           # 'yfinance', 'fred', 'manual', etc.
    "freq",             # 
    "market_region",    # 'US', 'Asia', 'Global' 등
    "unit",             # 예: 'USD', 'Index', 'Percent'
    "value",            # 원본 값
    "change_pct",       # 변화율
    "value_norm",       # 표준화된 값
    "retrieved_at"      # 데이터 수집 시각
]
