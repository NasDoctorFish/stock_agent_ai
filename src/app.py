# app.py
import streamlit as st
from core.db_handler import DBHandler
from core.db_updater import DBUpdater
from core.status_logger import StatusLogger
from core.decision_maker import DecisionMaker
from core.investor import Investor

from colorama import Fore, Style, init

from analysts.macro_analyst import MacroAnalyst
from analysts.fundamental_analyst import FundamentalAnalyst
from analysts.stock_data_analyst import StockDataAnalyst
from analysts.news_media_analyst import NewsMediaAnalyst
from analysts.business_analyst import BusinessAnalyst
from pathlib import Path
import pandas as pd

def main():
    st.title("📊 Macro Dashboard")

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

    if "df_feat" not in st.session_state:
        macro_result = macro.run()
        st.session_state.df_feat = macro_result.artifacts.get("df_feat")
        st.session_state.result = macro_result.data

    if st.session_state.df_feat is not None and macro.feat_cols:
        feature = st.selectbox("Feature", macro.feat_cols)
        fig = macro.visualize_macro_trend(st.session_state.df_feat, feature, True)
        if fig is not None:
            st.pyplot(fig)

if __name__ == "__main__":
    main()
