import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.dns_check import check_domain
from utils.explainability import explain_single_url
from utils.feature_extraction import FEATURE_NAMES
from utils.predictor import DEFAULT_THRESHOLD, BALANCED_THRESHOLD, HIGH_RECALL_THRESHOLD, predict_url
from utils.ssl_check import check_ssl
from utils.whois_check import get_domain_age


def show_single_scan():
    st.markdown(
        """
        <div class='main-title'>
        🔍 Single URL Scanner & Explainability
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("Analyze individual URLs with calibrated ML probabilities, risk assessment, and SHAP feature drivers.")
    st.write("")

    col_in, col_cfg = st.columns([2, 1])

    with col_in:
        url = st.text_input(
            "Enter URL to Inspect",
            placeholder="https://example.com/login"
        )

    with col_cfg:
        threshold_choice = st.selectbox(
            "Detection Threshold",
            [
                f"Default (Balanced F1: {DEFAULT_THRESHOLD:.2f})",
                f"High-Recall ({HIGH_RECALL_THRESHOLD:.2f}) - OOF Recall 95.1%; Holdout Recall 94.4%",
                "Custom Threshold"
            ]
        )
        if "Default" in threshold_choice:
            active_threshold = DEFAULT_THRESHOLD
        elif "High-Recall" in threshold_choice:
            active_threshold = HIGH_RECALL_THRESHOLD
        else:
            active_threshold = st.slider("Custom Threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.01)

    enable_network_checks = st.checkbox(
        "Enable live network validation (DNS, SSL, WHOIS with SSRF & timeout safeguards)",
        value=False,
        help="Optional live queries with strict 3-second timeouts and private-IP blocking."
    )

    scan = st.button("🚀 Scan URL", type="primary")

    if scan:
        if not url or not url.strip():
            st.warning("Please enter a valid URL")
            return

        clean_url = url.strip()
        if not clean_url.startswith(("http://", "https://")):
            clean_url = "http://" + clean_url

        # Fast progress feedback
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.003)
            progress.progress(i + 1)
        progress.empty()

        # Prediction via unified pipeline
        try:
            prediction, probability, features, risk_level = predict_url(clean_url, threshold=active_threshold)
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            return

        confidence = probability * 100

        # Optional live security checks
        dns_status = "Skipped"
        ssl_status = "Skipped"
        domain_age = "Skipped"

        if enable_network_checks:
            with st.spinner("Executing live security checks (DNS, SSL, WHOIS)..."):
                dns_res = check_domain(clean_url, timeout=3.0)
                dns_status = "Active" if dns_res else "Failed / Private IP"
                ssl_res = check_ssl(clean_url, timeout=3.0)
                ssl_status = "Valid" if ssl_res else "Invalid / Self-Signed"
                age_res = get_domain_age(clean_url, timeout=3.0)
                domain_age = f"{age_res} days" if age_res >= 0 else "Unknown / Private"

        # Result Cards
        st.write("")
        if prediction == 1:
            st.markdown(
                f"""
                <div class='phishing-card'>
                ⚠ PHISHING WEBSITE DETECTED &nbsp;|&nbsp; Risk Level: <b>{risk_level.upper()}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class='safe-card'>
                ✅ SAFE WEBSITE &nbsp;|&nbsp; Risk Level: <b>{risk_level.upper()}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Phishing Probability", f"{confidence:.1f}%")
        col2.metric("Risk Rating", risk_level)
        col3.metric("Operating Threshold", f"{active_threshold:.2f}")
        col4.metric("DNS Status", dns_status)
        col5.metric("SSL Status", ssl_status)

        # Threat Gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={"text": "Phishing Threat Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if prediction == 1 else "#10b981"},
                    "steps": [
                        {"range": [0, 35], "color": "rgba(16, 185, 129, 0.15)"},
                        {"range": [35, 70], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [70, 100], "color": "rgba(239, 68, 68, 0.15)"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": active_threshold * 100
                    }
                }
            )
        )
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # SHAP TOP 5 REASONS
        # -------------------------
        st.subheader("🧠 Top 5 Model Decision Drivers (SHAP Attribution)")
        st.caption("Local feature attributions indicating the strongest signals driving this specific prediction:")
        
        with st.spinner("Computing local SHAP explanations..."):
            reasons = explain_single_url(clean_url, top_k=5)

        if reasons:
            reasons_df = pd.DataFrame(reasons)
            reasons_df.columns = ["Feature Identifier", "Key Signal", "Feature Value", "SHAP Impact Score", "Signal Direction"]
            st.dataframe(
                reasons_df[["Key Signal", "Signal Direction", "SHAP Impact Score"]],
                use_container_width=True
            )
        else:
            st.info("SHAP explanation unavailable for this sample.")

        # -------------------------
        # Extracted Features Table
        # -------------------------
        st.subheader("📊 Extracted Lexical Features (40 Features)")
        names = FEATURE_NAMES if len(features) == len(FEATURE_NAMES) else [f"Feature {i+1}" for i in range(len(features))]
        formatted_vals = [round(v, 4) if isinstance(v, float) else v for v in features]

        feature_df = pd.DataFrame({
            "Feature Name": names,
            "Value": formatted_vals
        })
        st.dataframe(feature_df, use_container_width=True)