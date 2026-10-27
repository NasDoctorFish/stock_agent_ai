"""
Training-pool prices: current S&P 500 members NOT in config.yaml tickers,
2010 -> today, adjusted close, stored as table `price_pool` in
data/research/research.duckdb. The 56 config tickers stay the test/trade set.

Usage: poetry run python src/research/backfill_pool.py
"""
import io
import urllib.request

import duckdb
import pandas as pd
import yfinance as yf

from backfill_prices import ROOT, OUT_DB, load_tickers


def sp500() -> list:
    req = urllib.request.Request("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                                 headers={"User-Agent": "Mozilla/5.0"})
    t = pd.read_html(io.StringIO(urllib.request.urlopen(req, timeout=30).read().decode()))[0]
    return [s.replace(".", "-") for s in t.Symbol]


if __name__ == "__main__":
    pool = sorted(set(sp500()) - set(load_tickers()))
    raw = yf.download(pool, start="2010-01-01", auto_adjust=True, progress=False, threads=True)["Close"]
    px = raw.dropna(axis=1, thresh=252 * 3)  # drop tickers with <3y history
    long = px.rename_axis("date").reset_index().melt("date", var_name="stock_id", value_name="close").dropna()
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None)
    con = duckdb.connect(str(OUT_DB))
    con.register("p", long)
    con.execute("CREATE OR REPLACE TABLE price_pool AS SELECT * FROM p")
    print(f"pool: {px.shape[1]}/{len(pool)} tickers, {len(long)} rows, {long.date.min().date()} -> {long.date.max().date()}")
