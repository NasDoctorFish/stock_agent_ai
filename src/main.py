from core.db_handler import DBHandler
from core.db_updater import DBUpdater
from core.status_logger import StatusLogger
from core.decision_maker import DecisionMaker
from core.investor import Investor

from analysts.macro_analyst import MacroAnalyst
from analysts.fundamental_analyst import FundamentalAnalyst
from analysts.stock_data_analyst import StockDataAnalyst
from analysts.news_media_analyst import NewsMediaAnalyst
from analysts.business_analyst import BusinessAnalyst
from pathlib import Path
import pandas as pd


def main():
    logger = StatusLogger("System")
    db = DBHandler(logger)
    updater = DBUpdater(db, logger)

    # 업데이트
    logger.info("🔄 Updating data...")
    updater.update_csv_files() # complete
    logger.info("🔄 Saving to duckdb...")
    db.save_all_to_duckdb()

    # # 애널리스트 객체 생성
    macro = MacroAnalyst(db, logger)
    signal = macro.analyze()
    print(signal)
    # 애널리스트 데이터 시각화
    
    # ✅ base_dir은 항상 절대경로 기준으로
    BASE_DIR = Path(__file__).resolve().parents[1]   # "Stock ML" 폴더
    # logger = BASE_DIR)
    # MACRO_MODEL_PATH = BASE_DIR / "src" / "db" / "stockml.duckdb" #저장안됨
    df = macro.load_macro_data()
    macro.visualize_macro_trend(df)
    
    # fund = FundamentalAnalyst(db, logger)
    # stock = StockDataAnalyst(db, logger)
    # news = NewsMediaAnalyst(db, logger)
    # biz = BusinessAnalyst(db, logger)

    # # 투자자 객체 (모든 분석 종합)
    # investor = Investor([macro, fund, stock, news, biz], logger)

    # # 최종 의사결정
    # decision_maker = DecisionMaker(investor, logger)
    # decision_maker.run()

if __name__ == "__main__":
    main()
                                                                                                 