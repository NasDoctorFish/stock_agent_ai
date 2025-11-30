"""
DBUpdater
---------
Handles data ingestion, updating, and ETL before storing into DuckDB.
"""
from core.db_handler import DBHandler
import core.db_updater_utils as db_utils


class DBUpdater:
    def __init__(self, db: DBHandler, logger):
        self.db = db
        self.logger = logger
        self.load_settings_and_keys()
        
    # 🔽 utils 외부 함수 연결
    load_settings_and_keys = db_utils.load_settings_and_keys
    load_existing = db_utils.load_existing
    merge_new = db_utils.merge_new
    update_csv_files = db_utils.update_csv_files
    
    # def update_macro(self, df):
    #     self.db.insert_df(df, "macro_index_full")

    # def update_news(self, df):
    #     self.db.insert_df(df, "news_sentiment")
    
    # def update_all(self, **datasets):
    #     for name, df in datasets.items():
    #         self.db.insert_df(df, name)
