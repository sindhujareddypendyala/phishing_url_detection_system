"""
Bulk CSV Scanner Page.
Batch URL threat evaluation with formula-injection-safe export and distribution analytics.
"""

import io
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.predictor import predict_url
from utils.ui import (
    apply_custom_css,
    get_threshold_config,
    get_verdict_info,
    sanitize_csv_cell,
)

MAX_ROWS = 500


def show_bulk_scan_page():
    apply_custom_css()
    cfg = get_threshold_config()
    default_th = float(cfg.get("default_threshold", 0.52))
    high_rec_th = float(cfg.get("high_recall_threshold", 0.32))

    st.title("📂 Bulk URL Scanner")
    st.caption(f"Batch-process up to {MAX_ROWS} URLs from a CSV file with unified lexical threat analysis.")

    # Template download
    sample_df = pd.DataFrame({"url": [
        "https://www.google.com",
        "http://paypal.com.account-verify.xyz/login.php",
        "http://192.168.1.1/admin/login",
        "https://github.com/security",
    ]})
    st.download_button(
        "📥 Download Sample CSV Template",
        sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_urls_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV File (Requires 'url' column)", type=["csv"])

    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Failed to parse CSV file: {e}")
            return

        col_url = next((c for c in df_in.columns if c.strip().lower() == "url"), None)
        if col_url is None:
            st.error("❌ CSV must contain a column named 'url'. Found columns: " + ", ".join(df_in.columns))
            return

        raw_urls = df_in[col_url].dropna().astype(str).str.strip()
        raw_urls = [u for u in raw_urls if u]

        if not raw_urls:
            st.warning("⚠️ No valid URL entries found in the uploaded file.")
            return

        if len(raw_urls) > MAX_ROWS:
            st.warning(f"⚠️ Input contains {len(raw_urls)} URLs. Capping evaluation to first {MAX_ROWS} rows.")
            raw_urls = raw_urls[:MAX_ROWS]

        st.info(f"Loaded {len(raw_urls)} URLs ready for batch analysis.")

        if st.button("🚀 Process Batch Scan", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            for i, u in enumerate(raw_urls):
                try:
                    pred, prob, _, risk = predict_url(u, threshold=default_th)
                    verdict = get_verdict_info(prob, low_th=high_rec_th, high_th=default_th)
                    results.append({
                        "url": u,
                        "verdict": verdict["label"],
                        "threat_score": f"{prob*100:.1f}%",
                        "risk_level": risk,
                        "probability": round(prob, 4),
                        "status_code": pred,
                    })
                except Exception:
                    results.append({
                        "url": u,
                        "verdict": "Error",
                        "threat_score": "0.0%",
                        "risk_level": "Unknown",
                        "probability": 0.0,
                        "status_code": -1,
                    })

                progress_bar.progress((i + 1) / len(raw_urls))
                status_text.text(f"Processed {i+1} / {len(raw_urls)} URLs...")

            status_text.empty()
            progress_bar.empty()

            st.session_state.bulk_results_df = pd.DataFrame(results)

    # Display results if available
    if "bulk_results_df" in st.session_state and not st.session_state.bulk_results_df.empty:
        res_df = st.session_state.bulk_results_df
        total = len(res_df)
        phish_cnt = int((res_df["verdict"] == "Phishing").sum())
        susp_cnt = int((res_df["verdict"] == "Suspicious").sum())
        safe_cnt = int((res_df["verdict"] == "Safe").sum())

        st.subheader("📊 Batch Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scanned", f"{total:,}")
        c2.metric("🚨 Phishing Detected", f"{phish_cnt:,} ({(phish_cnt/total)*100:.1f}%)")
        c3.metric("⚠️ Suspicious URLs", f"{susp_cnt:,} ({(susp_cnt/total)*100:.1f}%)")
        c4.metric("✅ Safe URLs", f"{safe_cnt:,} ({(safe_cnt/total)*100:.1f}%)")

        # Distribution Chart
        chart_col, filter_col = st.columns([1, 1])
        with chart_col:
            dist_df = pd.DataFrame({
                "Category": ["Safe", "Suspicious", "Phishing"],
                "Count": [safe_cnt, susp_cnt, phish_cnt],
                "Color": ["#10b981", "#f59e0b", "#ef4444"],
            })
            fig = px.pie(
                dist_df,
                names="Category",
                values="Count",
                hole=0.45,
                color="Category",
                color_discrete_map={"Safe": "#10b981", "Suspicious": "#f59e0b", "Phishing": "#ef4444"},
                title="Threat Distribution",
            )
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

        with filter_col:
            st.subheader("🔍 Filter & Export")
            selected_filter = st.multiselect(
                "Filter by Verdict",
                options=["Safe", "Suspicious", "Phishing"],
                default=["Safe", "Suspicious", "Phishing"],
            )

        filtered_df = res_df[res_df["verdict"].isin(selected_filter)]
        st.dataframe(filtered_df[["url", "verdict", "risk_level", "threat_score"]], width="stretch", height=280)

        # Formula-injection safe export
        export_df = filtered_df.copy()
        export_df["url"] = export_df["url"].apply(sanitize_csv_cell)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Filtered Scan Results (CSV)",
            csv_bytes,
            file_name="cyber_shield_bulk_results.csv",
            mime="text/csv",
            type="primary",
        )


show_bulk_scan_page()

