import os
import json
import streamlit as st

def get_platform_metrics():
    default_metrics = {
        "total_urls": "11,256",
        "accuracy": "88.4%",
        "precision": "84.0%",
        "f1_score": "87.0%"
    }
    metrics_file = "reports/metrics.json"
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                h = data.get("operating_thresholds", {}).get("default_balanced", {})
                if not h:
                    best = data.get("best_model", "LightGBM")
                    h = data.get("models", {}).get(best, {}).get("holdout", {})
                return {
                    "total_urls": "11,256",
                    "accuracy": f"{h.get('accuracy', 0.8837)*100:.1f}%",
                    "precision": f"{h.get('precision', 0.8398)*100:.1f}%",
                    "f1_score": f"{h.get('f1', 0.8702)*100:.1f}%"
                }
        except Exception:
            pass
    return default_metrics

def show_home():

    st.markdown(
        """
        <div class='main-title'>
            🛡️ CYBERSHIELD AI
        </div>

        <div class='sub-title'>
            Enterprise Phishing Detection Platform
        </div>
        """,
        unsafe_allow_html=True
    )

    # ------------------------
    # HERO SECTION
    # ------------------------

    st.markdown(
        """
        <div class='cyber-card'>
            <h2>🚀 Welcome to CyberShield AI</h2>
            <p>
            Detect phishing websites using Machine Learning,
            DNS Intelligence, SSL Validation, and Threat Analytics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ------------------------
    # DASHBOARD METRICS
    # ------------------------

    metrics = get_platform_metrics()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>
                    Clean URLs
                </div>

                <div class='metric-value'>
                    {metrics['total_urls']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>
                    Accuracy (Holdout)
                </div>

                <div class='metric-value'>
                    {metrics['accuracy']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>
                    Precision
                </div>

                <div class='metric-value'>
                    {metrics['precision']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>
                    F1 Score
                </div>

                <div class='metric-value'>
                    {metrics['f1_score']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.write("")

    # ------------------------
    # FEATURES SECTION
    # ------------------------

    st.subheader("🔥 Platform Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class='cyber-card'>
            <h4>🔍 Single URL Detection</h4>

            Analyze websites instantly using
            Machine Learning and URL intelligence.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            """
            <div class='cyber-card'>
            <h4>📂 Bulk CSV Scanner</h4>

            Upload thousands of URLs and detect
            phishing websites automatically.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class='cyber-card'>
            <h4>📊 Analytics Dashboard</h4>

            Accuracy, Precision, Recall,
            F1 Score, and Confusion Matrix.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            """
            <div class='cyber-card'>
            <h4>⬇ Download Reports</h4>

            Export phishing detection results
            as CSV reports.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.write("")

    # ------------------------
    # ABOUT SECTION
    # ------------------------

    st.subheader("ℹ About")

    st.markdown(
        """
        CyberShield AI is a machine learning-based phishing
        detection platform designed to identify malicious
        URLs and improve cybersecurity awareness.

        Features include:

        - URL Analysis
        - Bulk Detection
        - Threat Analytics
        - CSV Report Export
        - DNS & SSL Validation
        """
    )