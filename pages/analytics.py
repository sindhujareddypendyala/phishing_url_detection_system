import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def load_metrics_report():
    metrics_path = "reports/metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def show_analytics():
    st.markdown(
        """
        <div class='main-title'>
        📊 Threat Analytics & Model Benchmark
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("Production model performance, group-aware cross-validation, and holdout evaluation.")
    st.write("")

    report = load_metrics_report()
    if report is None:
        st.warning("Metrics report not found. Run 'python train.py' to generate benchmark reports.")
        return

    best_name = report.get("best_model", "LightGBM")
    models_data = report.get("models", {})
    best_holdout = models_data.get(best_name, {}).get("holdout", {})
    best_cv = models_data.get(best_name, {}).get("cv_mean", {})
    best_cv_std = models_data.get(best_name, {}).get("cv_std", {})
    ds_info = report.get("dataset_info", {})

    # -------------------------
    # PRODUCTION MODEL METRICS
    # -------------------------
    st.subheader(f"🏆 Production Model: {best_name}")
    st.caption("Evaluated on 2,098 URLs across 1,327 unseen registered domains (zero domain overlap with training).")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Holdout Accuracy", f"{best_holdout.get('accuracy', 0)*100:.2f}%")
    col2.metric("Holdout Recall", f"{best_holdout.get('recall', 0)*100:.2f}%")
    col3.metric("Holdout Precision", f"{best_holdout.get('precision', 0)*100:.2f}%")
    col4.metric("Holdout F1 Score", f"{best_holdout.get('f1', 0)*100:.2f}%")
    col5.metric("Holdout ROC-AUC", f"{best_holdout.get('roc_auc', 0):.4f}")

    st.write("")

    # -------------------------
    # BENCHMARK COMPARISON TABLE
    # -------------------------
    st.subheader("📋 Grouped Cross-Validation & Holdout Benchmark")
    st.write(
        "All models trained and tested on identical registered domain groups using "
        "5-Fold StratifiedGroupKFold (on Train+Val) and GroupShuffleSplit holdout."
    )

    table_rows = []
    for m_name, m_stats in models_data.items():
        cv_f1 = f"{m_stats['cv_mean']['f1']*100:.2f}% ± {m_stats['cv_std']['f1']*100:.2f}%"
        h = m_stats["holdout"]
        table_rows.append({
            "Model": m_name,
            "5-Fold CV F1": cv_f1,
            "Holdout Accuracy": f"{h['accuracy']*100:.2f}%",
            "Holdout Precision": f"{h['precision']*100:.2f}%",
            "Holdout Recall": f"{h['recall']*100:.2f}%",
            "Holdout F1": f"{h['f1']*100:.2f}%",
            "Holdout ROC-AUC": f"{h['roc_auc']:.4f}",
            "Holdout PR-AUC": f"{h['pr_auc']:.4f}",
            "Holdout TN / FP / FN / TP": f"{h['tn']} / {h['fp']} / {h['fn']} / {h['tp']}"
        })

    benchmark_df = pd.DataFrame(table_rows)
    st.dataframe(benchmark_df, use_container_width=True)

    st.write("")

    # -------------------------
    # CONFUSION MATRIX
    # -------------------------
    col_cm, col_chart = st.columns(2)

    with col_cm:
        st.subheader("🎯 Holdout Confusion Matrix")
        tn = best_holdout.get("tn", 0)
        fp = best_holdout.get("fp", 0)
        fn = best_holdout.get("fn", 0)
        tp = best_holdout.get("tp", 0)

        cm_data = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual Safe (0)", "Actual Phishing (1)"],
            columns=["Pred Safe (0)", "Pred Phishing (1)"]
        )
        st.dataframe(cm_data, use_container_width=True)
        st.info(f"False Positives: {fp} | False Negatives: {fn} (out of {tn+fp+fn+tp} holdout samples)")

    with col_chart:
        st.subheader("📈 CV F1-Score Comparison")
        chart_data = pd.DataFrame({
            "Model": list(models_data.keys()),
            "CV F1 (%)": [m["cv_mean"]["f1"] * 100 for m in models_data.values()],
            "Holdout F1 (%)": [m["holdout"]["f1"] * 100 for m in models_data.values()]
        })
        fig = px.bar(
            chart_data,
            x="Model",
            y=["CV F1 (%)", "Holdout F1 (%)"],
            barmode="group",
            title="Cross-Validation vs Holdout F1-Score"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # DATASET OVERVIEW
    # -------------------------
    st.write("")
    st.subheader("📂 Dataset & Split Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Rows", f"{ds_info.get('raw_samples', 11430):,}")
    c2.metric("Clean Unique URLs", f"{ds_info.get('deduplicated_samples', 11256):,}")
    c3.metric("Registered Domains", f"{ds_info.get('unique_registered_domains', 6634):,}")
    c4.metric("Holdout URLs (Unseen Domains)", f"{ds_info.get('holdout_test_samples', 2098):,}")

    st.write("")
    st.subheader("📝 Dataset Sample Preview")
    try:
        df = pd.read_csv("data/phishingURL.csv")
        st.dataframe(df[["url", "status"]].head(15), use_container_width=True)
    except Exception:
        pass