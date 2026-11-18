"""
CyberShield AI - Main Application Entrypoint.
Streamlit 1.31+ st.navigation architecture with structured sidebar sections.
"""

import streamlit as st

st.set_page_config(
    page_title="CyberShield AI | Phishing URL Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Branding
st.sidebar.markdown(
    """
    <div style="padding: 10px 0 16px 0;">
        <h2 class="brand-name" style="margin: 0; display: flex; align-items: center; gap: 8px;">
            <span>🛡️</span> CyberShield AI
        </h2>
            <span style="font-size: 0.8rem; opacity: 0.75;">
            URL threat analysis
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation Definition using file paths
page_scan = st.Page("pages/scan_url.py", title="Scan URL", icon="🔍", default=True)
page_bulk = st.Page("pages/bulk_scan.py", title="Bulk Scan", icon="📂")
page_perf = st.Page("pages/model_performance.py", title="Model Performance", icon="📊")
page_about = st.Page("pages/about.py", title="About", icon="ℹ️")

nav = st.navigation({
    "Detection": [page_scan, page_bulk],
    "Platform": [page_perf, page_about]
})

nav.run()
