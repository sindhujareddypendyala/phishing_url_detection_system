import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import time

from utils.predictor import predict_url

# --------------------------
# PAGE
# --------------------------

def show_bulk_scan():

    st.markdown(
        """
        <div class='main-title'>
        📂 Bulk CSV Scanner
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Upload a CSV file containing URLs."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is None:
        return

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Error reading CSV: {e}"
        )

        return

    # --------------------------
    # VALIDATION
    # --------------------------

    if "url" not in df.columns:

        st.error(
            "CSV must contain a column named 'url'"
        )

        st.info(
            """
            Example:

            url
            https://google.com
            https://facebook.com
            """
        )

        return

    st.success(
        f"{len(df)} URLs Loaded"
    )

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    scan_button = st.button(
        "🚀 Start Bulk Scan"
    )

    if not scan_button:
        return

    # --------------------------
    # SCAN
    # --------------------------

    predictions = []
    confidences = []
    risk_levels = []

    progress = st.progress(0)

    total = len(df)

    for index, row in df.iterrows():

        url = str(row["url"])

        try:

            prediction, probability, features, risk_level = predict_url(url)

            confidence = round(
                probability * 100,
                2
            )

            pred_text = (
                "Phishing"
                if prediction == 1
                else "Safe"
            )

        except Exception:

            pred_text = "Error"
            confidence = 0
            risk_level = "Unknown"

        predictions.append(
            pred_text
        )

        confidences.append(
            round(
                confidence
            )
        )

        risk_levels.append(
            risk_level
        )

        progress.progress(
            int(
                ((index + 1) / total)
                * 100
            )
        )

    # --------------------------
    # RESULTS
    # --------------------------

    result_df = df.copy()

    result_df["Prediction"] = (
        predictions
    )

    result_df["Confidence (%)"] = (
        confidences
    )

    result_df["Risk Level"] = (
        risk_levels
    )

    safe_count = (
        result_df[
            result_df["Prediction"]
            == "Safe"
        ]
        .shape[0]
    )

    phishing_count = (
        result_df[
            result_df["Prediction"]
            == "Phishing"
        ]
        .shape[0]
    )

    # --------------------------
    # SUMMARY CARDS
    # --------------------------

    st.write("")
    st.subheader(
        "📊 Scan Summary"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total URLs",
        len(result_df)
    )

    col2.metric(
        "Safe URLs",
        safe_count
    )

    col3.metric(
        "Phishing URLs",
        phishing_count
    )

    # --------------------------
    # CHART
    # --------------------------

    chart_df = pd.DataFrame({

        "Category": [
            "Safe",
            "Phishing"
        ],

        "Count": [
            safe_count,
            phishing_count
        ]
    })

    fig = px.pie(
        chart_df,
        values="Count",
        names="Category",
        title="Threat Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------
    # RESULTS TABLE
    # --------------------------

    st.subheader(
        "📋 Detection Results"
    )

    st.dataframe(
        result_df,
        use_container_width=True
    )

    # --------------------------
    # DOWNLOAD
    # --------------------------

    csv = result_df.to_csv(
        index=False
    )

    st.download_button(

        label="⬇ Download Results CSV",

        data=csv,

        file_name=
        "cybershield_results.csv",

        mime="text/csv"
    )

    st.success(
        "Bulk Scan Completed Successfully"
    )