"""
app.py
------
Entry point of the Streamlit app. It only handles page setup and navigation;
each page lives in its own file inside the src folder.

Run with:  python -m streamlit run app.py
"""

import streamlit as st

from src.ui_about import show_about
from src.ui_common import apply_theme
from src.ui_home import show_home
from src.ui_performance import show_performance
from src.ui_scanner import show_scanner

st.set_page_config(
    page_title="PhishGuard - Phishing URL Detection",
    page_icon="🛡️",
    layout="wide",
)
apply_theme()

PAGES = {
    "Home": show_home,
    "URL Scanner": show_scanner,
    "Model Performance": show_performance,
    "About Project": show_about,
}

st.sidebar.markdown("## 🛡️ PhishGuard")
st.sidebar.caption("Explainable ML for phishing URL detection")
choice = st.sidebar.radio("Navigation", list(PAGES.keys()), key="nav")
st.sidebar.divider()
st.sidebar.caption("College mini-project. Predictions are estimates, not guarantees of safety.")

PAGES[choice]()