"""
Build a research DuckDB with long price history (default 2010 -> today).

The live DB (src/db/stockml.duckdb) is never written: the daemon holds its
write lock. We snapshot it (+WAL), replace `price_daily_full` with a long
adjusted-OHLCV history for config.yaml tickers, and write the result to
data/research/research.duckdb. Point the backtester at it with --db.

yfinance history() defaults to auto_adjust=True, so close/open/high/low are
split- and dividend-adjusted (total-return basis).

Usage: python src/research/backfill_prices.py [--start 2010-01-01]
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
LIVE_DB = ROOT / "src" / "db" / "stockml.duckdb"
OUT_DB = ROOT / "data" / "research" / "research.duckdb"


def load_tickers() -> list:
    cfg = yaml.safe_load(open(ROOT / "src" / "config" / "config.yaml"))
    return sorted(set(cfg["tickers"]))


def download(tickers: list, start: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, auto_adjust=True, group_by="ticker",
                      threads=True, progress=False)
    frames = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            print(f"no data: {t}")
            continue
        d = raw[t].dropna(subset=["Close"]).reset_index()
        d.columns = [str(c).lower() for c in d.columns]
        d["stock_id"] = t
        frames.append(d[["date", "open", "high", "low", "close", "volume", "stock_id"]])
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[(df["close"] > 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    args = ap.parse_args()

    tickers = load_tickers()
    df = download(tickers, args.start)
    print(f"{len(df)} rows, {df.stock_id.nunique()}/{len(tickers)} tickers, "
          f"{df.date.min().date()} -> {df.date.max().date()}")

    tmp = Path(tempfile.mkdtemp()) / "snap.duckdb"
    shutil.copy2(LIVE_DB, tmp)
    wal = Path(f"{LIVE_DB}.wal")
    if wal.exists():
        shutil.copy2(wal, f"{tmp}.wal")

    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(tmp))
    con.register("p", df)
    con.execute("DROP TABLE IF EXISTS price_daily_full")
    con.execute("CREATE TABLE price_daily_full AS SELECT * FROM p")
    con.execute("CHECKPOINT")
    con.close()
    shutil.move(str(tmp), OUT_DB)
    print(f"wrote {OUT_DB}")


if __name__ == "__main__":
    main()
