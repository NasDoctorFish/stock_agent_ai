import duckdb

conn = duckdb.connect("src/db/stockml.duckdb")
print(conn.execute("SHOW TABLES").fetchall())
print(conn.execute("DESCRIBE macro_index_full").fetchdf())
print(conn.execute("SELECT * FROM macro_index_full LIMIT 5;").fetchdf())
print(conn.execute("SELECT * FROM news_sentiment LIMIT 5;").fetchdf())
print(conn.execute("SELECT * FROM price_daily_full LIMIT 5;").fetchdf())
print(conn.execute("SELECT * FROM fred_series_catalog LIMIT 5;").fetchdf())
print(conn.execute("SELECT * FROM fundamentals_q_full LIMIT 5;").fetchdf())

print(
    conn.execute(
        """SELECT
  column_name,
  data_type,
  column_default,
  is_nullable
FROM information_schema.columns
WHERE table_name = 'macro_index_full';
"""
    ).fetchdf()
)



conn = duckdb.connect("data/dart.duckdb")
print(conn.execute("SHOW TABLES").fetchall())
print(conn.execute("DESCRIBE dart_filings").fetchdf())

print(conn.execute("SELECT corp_code, corp_name, rcept_no, report_name, report_date FROM dart_filings LIMIT 5;").fetchdf())
# poetry run python3 sql/test.py
