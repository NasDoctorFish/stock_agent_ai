from pathlib import Path

# ===========================
# 📁 폴더 및 파일 구조 정의
# ===========================
structure = {
    "core": [
        "db_handler.py",
        "db_updater.py",
        "status_logger.py",
        "decision_maker.py",
        "investor.py",
    ],
    "analysts": [
        "base_analyst.py",
        "news_media_analyst.py",
        "fundamental_analyst.py",
        "macro_analyst.py",
        "stock_data_analyst.py",
        "business_analyst.py",
    ],
}

# ===========================
# 🧱 기본 템플릿 정의
# ===========================
TEMPLATES = {
    "core/db_handler.py": '''"""
DBHandler
---------
Handles low-level DuckDB I/O operations.
"""
import duckdb
from pathlib import Path

class DBHandler:
    def __init__(self, db_path: str = "data/stockml.duckdb"):
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
        print(f"🦆 Connected to DuckDB at {self.db_path}")

    def execute(self, query: str):
        return self.conn.execute(query)

    def insert_df(self, df, table_name: str):
        self.conn.register("df", df)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df LIMIT 0;")
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM df;")
        print(f"✅ Inserted {len(df)} rows into {table_name}")

    def close(self):
        self.conn.close()
        print("🔒 Connection closed.")
''',

    "core/db_updater.py": '''"""
DBUpdater
---------
Handles data ingestion, updating, and ETL before storing into DuckDB.
"""
from core.db_handler import DBHandler

class DBUpdater:
    def __init__(self, db: DBHandler):
        self.db = db

    def update_macro(self, df):
        self.db.insert_df(df, "macro_index_full")

    def update_news(self, df):
        self.db.insert_df(df, "news_sentiment")

    def update_all(self, **datasets):
        for name, df in datasets.items():
            self.db.insert_df(df, name)
''',

    "core/status_logger.py": '''"""
StatusLogger
------------
Manages system-wide logging and monitoring.
"""
import logging
from datetime import datetime

class StatusLogger:
    def __init__(self, name="system"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def info(self, msg): self.logger.info(msg)
    def warn(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
''',

    "core/decision_maker.py": '''"""
DecisionMaker
-------------
Integrates multiple analysts and produces actionable investment decisions.
"""
class DecisionMaker:
    def __init__(self, analysts: list):
        self.analysts = analysts

    def make_decision(self, market_state):
        results = {a.__class__.__name__: a.analyze(market_state) for a in self.analysts}
        # Basic rule-based example
        sentiment = results.get("NewsMediaAnalyst", {}).get("sentiment", 0)
        macro = results.get("MacroAnalyst", {}).get("macro_trend", "neutral")

        if macro == "bullish" and sentiment > 0.5:
            return "BUY"
        elif macro == "bearish" and sentiment < -0.5:
            return "SELL"
        else:
            return "HOLD"
''',

    "core/investor.py": '''"""
Investor
--------
Top-level orchestrator that uses the DecisionMaker and Analysts to act.
"""
from core.status_logger import StatusLogger

class Investor:
    def __init__(self, decision_maker, logger: StatusLogger):
        self.brain = decision_maker
        self.logger = logger

    def act(self, market_state):
        decision = self.brain.make_decision(market_state)
        self.logger.info(f"Decision: {decision}")
        return decision
''',

    "analysts/base_analyst.py": '''"""
BaseAnalyst
-----------
Abstract interface for all analyst classes.
"""
from abc import ABC, abstractmethod

class BaseAnalyst(ABC):
    @abstractmethod
    def analyze(self, data):
        pass
''',

    "analysts/news_media_analyst.py": '''from analysts.base_analyst import BaseAnalyst

class NewsMediaAnalyst(BaseAnalyst):
    def analyze(self, data):
        # Example placeholder
        return {"sentiment": 0.7, "confidence": 0.8}
''',

    "analysts/fundamental_analyst.py": '''from analysts.base_analyst import BaseAnalyst

class FundamentalAnalyst(BaseAnalyst):
    def analyze(self, data):
        return {"valuation": "undervalued", "pe_ratio": 12.5}
''',

    "analysts/macro_analyst.py": '''from analysts.base_analyst import BaseAnalyst

class MacroAnalyst(BaseAnalyst):
    def analyze(self, data):
        return {"macro_trend": "bullish", "inflation": 2.4}
''',

    "analysts/stock_data_analyst.py": '''from analysts.base_analyst import BaseAnalyst

class StockDataAnalyst(BaseAnalyst):
    def analyze(self, data):
        return {"momentum": 0.65, "volatility": 0.22}
''',

    "analysts/business_analyst.py": '''from analysts.base_analyst import BaseAnalyst

class BusinessAnalyst(BaseAnalyst):
    def analyze(self, data):
        return {"industry_outlook": "positive", "competition": "moderate"}
''',
}

# ===========================
# 🛠️ 파일 생성 로직
# ===========================
def create_files(base_dir="src"):
    base = Path(base_dir)
    for folder, files in structure.items():
        dir_path = base / folder
        dir_path.mkdir(parents=True, exist_ok=True)
        for filename in files:
            file_path = dir_path / filename
            template_key = f"{folder}/{filename}"
            content = TEMPLATES.get(template_key, "# empty\n")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Created {file_path}")

if __name__ == "__main__":
    create_files()
