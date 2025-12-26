import duckdb

conn = duckdb.connect("src/db/stockml.duckdb")
print(conn.execute("SHOW TABLES").fetchall())
print(conn.execute("DESCRIBE macro_index_full").fetchdf())
print(conn.execute("SELECT * FROM macro_index_full LIMIT 5;").fetchdf())
print(conn.execute("SELECT * FROM news_sentiment LIMIT 5;").fetchdf())
print(conn.execute("SELECT * FROM price_daily_full LIMIT 5;").fetchdf())
print(conn.execute("SELECT * FROM fred_series_catalog LIMIT 5;").fetchdf())
print(conn.execute("SELECT * FROM fundamentals_q_full LIMIT 5;").fetchdf())
