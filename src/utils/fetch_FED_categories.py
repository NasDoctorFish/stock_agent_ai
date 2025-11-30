import requests
import pandas as pd
import time
from tqdm import tqdm

API_KEY = "9e2a2fdc145365f860c4fae5e1f7d4b7"
BASE_URL = "https://api.stlouisfed.org/fred"

def get_series_in_category(cat_id):
    """특정 카테고리 안의 시리즈 목록"""
    url = f"{BASE_URL}/category/series"
    params = {"category_id": cat_id, "api_key": API_KEY, "file_type": "json"}
    res = requests.get(url, params=params, timeout=15)
    if res.status_code != 200:
        return pd.DataFrame()
    data = res.json().get("seriess", [])
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)[["id", "title", "frequency", "units", "seasonal_adjustment_short"]].assign(category_id=cat_id)

def get_child_categories(cat_id):
    """특정 카테고리의 하위 카테고리 ID 목록"""
    url = f"{BASE_URL}/category/children"
    params = {"category_id": cat_id, "api_key": API_KEY, "file_type": "json"}
    res = requests.get(url, params=params, timeout=15)
    if res.status_code != 200:
        return []
    return [c["id"] for c in res.json().get("categories", [])]

def crawl_fred_series(cat_id=0, depth=0, max_depth=3, sleep=0.7):
    """카테고리 트리 전체 탐색해서 모든 시리즈 DataFrame으로 반환"""
    dfs = []
    df_series = get_series_in_category(cat_id)
    if not df_series.empty:
        dfs.append(df_series)
    if depth < max_depth:
        children = get_child_categories(cat_id)
        for child_id in tqdm(children, desc=f"Depth {depth} → {depth+1} (cat {cat_id})", leave=False):
            time.sleep(sleep)
            child_df = crawl_fred_series(child_id, depth+1, max_depth)
            if not child_df.empty:
                dfs.append(child_df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

# 실행
fred_df = crawl_fred_series(max_depth=3)
print(f"✅ 수집 완료: {len(fred_df)} series")
print(fred_df.head(10))

# CSV로 저장
fred_df.to_csv("fred_series_catalog.csv", index=False)

# DuckDB에 삽입
import duckdb
con = duckdb.connect("src/db/stockml.duckdb")
con.execute("CREATE TABLE IF NOT EXISTS fred_series_catalog AS SELECT * FROM fred_df")

# Duck DB testing
import duckdb
con = duckdb.connect()
df = con.execute("SELECT * FROM 'data/raw/fred_series_catalog.csv' LIMIT 10").df()
print(df)