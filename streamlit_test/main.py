import streamlit as st
from pathlib import Path

# Path
# streamlit_test/main.py

BASE_DIR = Path(__file__).resolve().parents[1]   # "Stock ML" 폴더

st.title("MARK STOCK II by JU")

st.title("MARK STOCK II by JU")
st.write("Your all-in-one stock analysis and visualization tool.")


# links to pages
st.page_link("pages/1_Mortgage.py", label="Mortgage", icon="🏠")
st.page_link("pages/2_stock_view.py", label="Stock Analysis", icon="📈")
st.page_link("pages/3_huggingface.py", label="Hugging Face", icon="🌐")