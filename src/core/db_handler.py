"""
DBHandler
---------
Handles low-level DuckDB I/O operations.

Usage Example
-------------
from core.db_handler import DBHandler
from core.status_logger import StatusLogger

logger = StatusLogger("system.log")
db = DBHandler(logger)

# ✅ Execute custom query
df = db.execute("SELECT * FROM price_daily_full LIMIT 5", fetch=True)
print(df)

# ✅ Load CSV files directly into DuckDB tables
db.load_csv_to_duckdb("data/raw/price_daily.csv", "price_daily_full")

# ✅ Automatically load all CSVs (if exists)
db.save_all_to_duckdb()

# ✅ Insert a DataFrame directly (append mode)
db.insert_dataframe(df, "price_daily_full")

# ✅ Export a table to Parquet
db.export_parquet("price_daily_full", "data/export/price_daily.parquet")

# ✅ Check if a table exists
exists = db.table_exists("price_daily_full")
print("Exists:", exists)

db.close()

"""

import duckdb
from pathlib import Path
from core.status_logger import StatusLogger

print(Path(__file__))


class DBHandler:
    def __init__(self, logger):
        self.base_dir = Path(__file__).resolve().parents[2]
        self.db_path = self.base_dir / "src" / "db" / "stockml.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        self.logger = logger
        self.logger.info(f"🦆 Connected to DuckDB at {self.db_path}")

    def execute(self, query: str, fetch: bool = False):
        try:
            self.logger.info(f"Executing query: {query[:60]}...")
            result = self.conn.execute(query)
            return result.df() if fetch else result
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return None

    def query(self, sql: str):
        """DuckDB 쿼리 결과를 DataFrame으로 반환"""
        return self.conn.execute(sql).fetchdf()


    def load_csv_to_duckdb(self, csv_path, table_name):
        """CSV 파일을 DuckDB 테이블로 로드 (덮어쓰기 모드)."""
        try:
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT * FROM read_csv_auto('{csv_path}');
            """)
            self.logger.info(f"✅ {csv_path} → {table_name} 저장 완료")
        except Exception as e:
            self.logger.error(f"❌ {csv_path} 로드 실패: {e}")

    def save_all_to_duckdb(self):
        '''
        load_csv_to_duckdb
        only if the csv file exists in data/raw
        '''
        
        csv_to_table = {
            "fundamentals_quarterly.csv": "fundamentals_q_full",
            "macro_index.csv": "macro_index_full",
            "news_sentiment.csv": "news_sentiment",
            "price_daily.csv": "price_daily_full"
        }

        for filename, table_name in csv_to_table.items():
            path = self.base_dir / "data" / "raw" / filename
            if not path.exists():
                print(f"⚠️ File not found: {path}")
                continue

            self.load_csv_to_duckdb(path, table_name)
            self.logger.info(f"LOADING COMPLETED: {table_name} in {path}")
        self.logger.info("Done!")

    # ✅ 추가 함수들
    def insert_dataframe(self, df, table_name):
        """Pandas DataFrame을 DuckDB 테이블에 추가 (append)."""
        try:
            self.conn.register("temp_df", df)
            self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
            self.conn.unregister("temp_df")
            self.logger.info(f"✅ Inserted {len(df)} rows → {table_name}")
        except Exception as e:
            self.logger.error(f"❌ DataFrame insert 실패 ({table_name}): {e}")

    def export_parquet(self, table_name, output_path):
        """DuckDB 테이블을 Parquet 파일로 내보내기."""
        try:
            self.conn.execute(f"COPY {table_name} TO '{output_path}' (FORMAT PARQUET)")
            self.logger.info(f"📦 Exported {table_name} → {output_path}")
        except Exception as e:
            self.logger.error(f"❌ Export 실패 ({table_name}): {e}")

    def table_exists(self, table_name):
        """테이블 존재 여부 확인."""
        try:
            result = self.conn.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name='{table_name}'
            """).fetchone()[0]
            return result > 0
        except Exception as e:
            self.logger.error(f"❌ Table existence check 실패 ({table_name}): {e}")
            return False

    def close(self):
        self.conn.close()
        self.logger.info("🔒 DuckDB connection closed.")
