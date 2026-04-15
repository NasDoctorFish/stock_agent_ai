# %%
"""
DataFetcher.py — Exploratory data fetching using Alpha Vantage.
For the full regime detector, see regime_detector.py.
"""
import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv("../.env")

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # replace with your key from https://www.alphavantage.co/support/#api-key
BASE_URL = "https://www.alphavantage.co/query"


# ── Weekly Price Data ─────────────────────────────────────────────────────────
# %%
def fetch_weekly_prices(symbol: str, api_key: str = API_KEY) -> pd.DataFrame:
    """Fetch weekly OHLCV data for a symbol from Alpha Vantage."""
    url = f"{BASE_URL}?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json()

    weekly = data["Weekly Time Series"]
    df = pd.DataFrame.from_dict(weekly, orient="index")
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df.sort_index()


# %%
spy = fetch_weekly_prices("SPY")
print(spy.tail())


# ── Treasury Yields (Yield Curve) ─────────────────────────────────────────────
# %%
def fetch_treasury_yield(maturity: str = "10year", api_key: str = API_KEY) -> pd.Series:
    """Fetch monthly Treasury yield for the given maturity."""
    url = f"{BASE_URL}?function=TREASURY_YIELD&interval=monthly&maturity={maturity}&apikey={api_key}"
    data = requests.get(url).json()

    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).set_index("date").sort_index()
    return df["value"].rename(maturity)


# %%
y10 = fetch_treasury_yield("10year")
y2  = fetch_treasury_yield("2year")
spread = y10 - y2
print("Yield curve spread (10yr - 2yr):")
print(spread.tail())


# ── VIX ───────────────────────────────────────────────────────────────────────
# %%
vix = fetch_weekly_prices("VIX")
print("\nVIX (close):")
print(vix["close"].tail())


# ── Credit Spread Proxy (HYG) ─────────────────────────────────────────────────
# %%
hyg = fetch_weekly_prices("HYG")
print("\nHYG (High Yield ETF close):")
print(hyg["close"].tail())


# ── Liquidity (Federal Funds Rate) ────────────────────────────────────────────
# %%
def fetch_federal_funds_rate(api_key: str = API_KEY) -> pd.Series:
    """Fetch monthly Federal Funds Rate from Alpha Vantage."""
    url = f"{BASE_URL}?function=FEDERAL_FUNDS_RATE&interval=monthly&apikey={api_key}"
    data = requests.get(url).json()

    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).set_index("date").sort_index()
    return df["value"].rename("fed_funds_rate")


# %%
ffr = fetch_federal_funds_rate()
print("\nFederal Funds Rate:")
print(ffr.tail())


# ── Momentum (computed from SPY) ──────────────────────────────────────────────
# %%
ret_12m  = spy["close"].pct_change(52)
ret_1m   = spy["close"].pct_change(4)
momentum = ret_12m - ret_1m
print("\nSPY Momentum (12m - 1m):")
print(momentum.tail())


# ── Quick Regime Snapshot ─────────────────────────────────────────────────────
# %%
# For the full regime detection pipeline, run:
#   python regime_detector.py --api-key YOUR_KEY
