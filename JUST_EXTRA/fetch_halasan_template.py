import yfinance as yf
import pandas as pd
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from IPython.display import display
import os


def cagr(start, end, years=5):
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def get_5yr_growth(ticker, item):
    try:
        stmt = ticker.financials
        vals = stmt.loc[item].sort_index()
        if len(vals) < 5:
            return None
        return cagr(vals.iloc[0], vals.iloc[-1])
    except:
        return None


def process_company(company):
    name = company["name"]
    ticker_code = company["ticker"]

    try:
        ticker = yf.Ticker(ticker_code)
        info = ticker.info

        price = info.get("currentPrice")
        dividend = info.get("dividendRate")
        dividend_yield = (info.get("dividendYield") or 0) * 100
        market_cap = info.get("marketCap")
        per = info.get("trailingPE")
        roe = (info.get("returnOnEquity") or 0) * 100
        pbr = info.get("priceToBook")
        wk52 = f"{info.get('fiftyTwoWeekLow')} ~ {info.get('fiftyTwoWeekHigh')}"

        rev_cagr = get_5yr_growth(ticker, "Total Revenue")
        op_cagr  = get_5yr_growth(ticker, "Operating Income")

        earnings_yield = (1 / per) * 100 if per else None

        score = (
            (roe or 0) * 0.4 +
            (earnings_yield or 0) * 0.3 +
            ((rev_cagr or 0) * 100) * 0.3
        )

        return {
            "기업명": name,
            "티커": ticker_code,
            "스코어 ▼": round(score, 2),
            "5년 연평균 매출액 성장률": f"{rev_cagr*100:.2f}%" if rev_cagr else None,
            "5년 연평균 영업이익 성장률": f"{op_cagr*100:.2f}%" if op_cagr else None,
            "주가 ↕": price,
            "배당금 ↕": dividend,
            "배당 수익률(%)": dividend_yield,
            "시총 (억원)": round((market_cap or 0)/1e8, 2) if market_cap else None,
            "PER(배)": per,
            "이익수익률(%)": earnings_yield,
            "PBR(배)": pbr,
            "ROE (%)": roe,
            "52주 가격범위": wk52
        }

    except Exception as e:
        print(f"{name} ({ticker_code}) 에러: {e}")
        return None


def generate_csv_from_yaml(yaml_file="companies.yaml", output="stocks.csv"):
    with open(yaml_file, "r", encoding="utf-8") as f:
        companies = yaml.safe_load(f)["companies"]

    results = []

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_company, comp): comp for comp in companies}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"완료! '{output}' 생성됨.")
    return df


if __name__ == "__main__":
    # ✅ base_dir은 항상 절대경로 기준으로
    BASE_DIR = Path(__file__).resolve().parents[1]   # "Stock ML" 폴더
    FILE_PATH = BASE_DIR / "JUST_EXTRA" / "companies.yaml"
    OUTPUT_PATH = BASE_DIR / "JUST_EXTRA" / "stocks_halasan.csv"
    # generate_csv_from_yaml(yaml_file = FILE_PATH, output = OUTPUT_PATH)
    
    df = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
    display(df[:10])


### yfincne의 한계 나중에 KRX 이용하기