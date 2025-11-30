import duckdb

conn = duckdb.connect("src/db/stockml.duckdb")
print(conn.execute("SHOW TABLES").fetchall())
print(conn.execute("DESCRIBE macro_index_full").fetchdf())
print(conn.execute("SELECT * FROM macro_index_full LIMIT 5;").fetchdf())