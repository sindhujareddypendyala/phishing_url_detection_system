import streamlit as st

from pages.home import show_home
from pages.single_scan import show_single_scan
from pages.bulk_scan import show_bulk_scan
from pages.analytics import show_analytics

# ------------------------
# PAGE CONFIG
# ------------------------

st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------
# LOAD CSS
# ------------------------

with open("assets/styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ------------------------
# SIDEBAR
# ------------------------

st.sidebar.markdown(
    """
    # 🛡️ CyberShield AI

    Advanced Phishing Detection Platform
    """
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Single URL Scan",
        "📂 Bulk CSV Scan",
        "📊 Analytics"
    ]
)

# ------------------------
# ROUTING
# ------------------------

if page == "🏠 Home":
    show_home()

elif page == "🔍 Single URL Scan":
    show_single_scan()

elif page == "📂 Bulk CSV Scan":
    show_bulk_scan()

elif page == "📊 Analytics":
    show_analytics()