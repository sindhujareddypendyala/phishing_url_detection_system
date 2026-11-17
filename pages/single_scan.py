import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

from utils.predictor import predict_url
from utils.dns_check import check_domain
from utils.ssl_check import check_ssl
from utils.whois_check import get_domain_age


def show_single_scan():

    st.markdown(
        """
        <div class='main-title'>
        🔍 Single URL Scanner
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    url = st.text_input(
        "Enter URL",
        placeholder="https://example.com"
    )

    scan = st.button("🚀 Scan URL")

    if scan:

        if not url:

            st.warning(
                "Please enter a URL"
            )
            return

        if not url.startswith(
            ("http://", "https://")
        ):

            st.error(
                "URL must start with http:// or https://"
            )
            return

        # -------------------------
        # Scan Animation
        # -------------------------

        progress = st.progress(0)

        for i in range(100):

            time.sleep(0.01)

            progress.progress(i + 1)

        # -------------------------
        # Prediction
        # -------------------------

        try:

            prediction, probability, features = (
                predict_url(url)
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )
            return

        confidence = probability * 100

        # -------------------------
        # Security Checks
        # -------------------------

        dns_status = check_domain(url)

        ssl_status = check_ssl(url)

        domain_age = get_domain_age(url)

        # -------------------------
        # Result Card
        # -------------------------

        st.write("")

        if prediction == 1:

            st.markdown(
                """
                <div class='phishing-card'>
                ⚠ PHISHING WEBSITE DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class='safe-card'>
                ✅ SAFE WEBSITE
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # -------------------------
        # Metrics
        # -------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        col2.metric(
            "DNS Status",
            "Active" if dns_status else "Failed"
        )

        col3.metric(
            "SSL Status",
            "Valid" if ssl_status else "Invalid"
        )

        col4.metric(
            "Domain Age",
            str(domain_age)
        )

        # -------------------------
        # Threat Gauge
        # -------------------------

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={
                    "text":
                    "Threat Score"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    }
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------
        # Security Logs
        # -------------------------

        st.subheader(
            "🖥 Security Logs"
        )

        st.code(
f"""
[INFO] URL Received

[INFO] DNS Status:
{dns_status}

[INFO] SSL Status:
{ssl_status}

[INFO] Domain Age:
{domain_age}

[INFO] Prediction:
{'Phishing' if prediction == 1 else 'Safe'}

[INFO] Confidence:
{confidence:.2f}%

[INFO] Scan Completed Successfully
"""
        )

        # -------------------------
        # Feature Table
        # -------------------------

        st.subheader(
            "📊 Extracted Features"
        )

        feature_df = pd.DataFrame({

            "Feature No":
            range(
                1,
                len(features) + 1
            ),

            "Value":
            features
        })

        st.dataframe(
            feature_df,
            use_container_width=True
        )