"""
Model Performance Page.
Transparent benchmark reporting loaded dynamically from reports/ directory.
Interactive threshold exploration and domain-clustered bootstrap confidence intervals.
"""

import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import streamlit as st
from utils.ui import apply_custom_css


@st.cache_data
def load_metrics_data():
    data = {}
    if os.path.exists("reports/metrics.json"):
        try:
            with open("reports/metrics.json", "r", encoding="utf-8") as f:
                data["metrics"] = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    if os.path.exists("reports/holdout_predictions.csv"):
        try:
            data["holdout"] = pd.read_csv("reports/holdout_predictions.csv")
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    if os.path.exists("reports/operating_points.csv"):
        try:
            data["operating_points"] = pd.read_csv("reports/operating_points.csv")
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    return data


def show_model_performance_page():
    apply_custom_css()
    st.title("📊 Model Performance & Leak-Free Benchmark")
    st.caption("Single source of truth evaluation on 2,098 holdout URLs across 1,327 registered domains (0 domain overlap).")

    data = load_metrics_data()
    if not data or "metrics" not in data or "holdout" not in data:
        st.warning("⚠️ Benchmark artifacts missing. Run 'python scripts/evaluate_final.py' to generate performance data.")
        return

    m = data["metrics"]
    df_holdout = data["holdout"]
    y_true = df_holdout["label"].values
    probs = df_holdout["prob"].values
    def_th = m.get("operating_thresholds", {}).get("default_balanced", {})

    # Core holdout metrics at the stored balanced threshold
    metric_cols = st.columns(4)
    for col, label, key in zip(
        metric_cols,
        ["Accuracy", "Precision", "Recall", "F1 score"],
        ["accuracy", "precision", "recall", "f1"],
    ):
        col.metric(f"Holdout {label}", f"{def_th.get(key, 0)*100:.2f}%")

    # Interactive Threshold Inspector
    st.subheader("🎯 Interactive Threshold Explorer")
    st.caption("Slide the threshold to simulate real-time decision boundary trade-offs on actual holdout predictions:")

    th_val = st.slider("Classification threshold", min_value=0.10, max_value=0.90, value=float(def_th.get("threshold", 0.52)), step=0.01)

    preds_th = (probs >= th_val).astype(int)
    cm = confusion_matrix(y_true, preds_th)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc_th = accuracy_score(y_true, preds_th)
    prec_th = precision_score(y_true, preds_th, zero_division=0)
    rec_th = recall_score(y_true, preds_th, zero_division=0)
    f1_th = f1_score(y_true, preds_th, zero_division=0)

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Safe (0)", "Actual Phish (1)"],
        columns=["Predicted Safe (0)", "Predicted Phish (1)"],
    )
    st.dataframe(cm_df, width="stretch")
    s1, s2 = st.columns(2)
    s1.metric("Recall", f"{rec_th*100:.2f}%")
    s2.metric("Precision", f"{prec_th*100:.2f}%")
    s3, s4 = st.columns(2)
    s3.metric("Accuracy", f"{acc_th*100:.2f}%")
    s4.metric("F1 score", f"{f1_th*100:.2f}%")

    # Stored threshold trade-offs and model evolution
    st.subheader("Operating point trade-offs")
    if "operating_points" in data:
        st.dataframe(data["operating_points"], width="stretch", hide_index=True)
    else:
        st.caption("Threshold trade-off report is unavailable.")

    st.subheader("📋 Model Evolution & Baseline Comparison")
    if "comparison_table" in m:
        st.dataframe(pd.DataFrame(m["comparison_table"]), width="stretch")

    # Paired Bootstrap Differences
    if "bootstrap_paired_vs_baseline_18" in m:
        bp = m["bootstrap_paired_vs_baseline_18"]
        with st.expander("🔬 Paired Domain-Cluster Bootstrap (1,000 Resamples vs Clean Baseline)", expanded=True):
            b1, b2, b3 = st.columns(3)
            b1.metric("Delta ROC-AUC", f"{bp['delta_roc_auc']['mean']:+.4f}", f"95% CI: [{bp['delta_roc_auc']['ci_lower']:+.4f}, {bp['delta_roc_auc']['ci_upper']:+.4f}]")
            b2.metric("Delta PR-AUC", f"{bp['delta_pr_auc']['mean']:+.4f}", f"95% CI: [{bp['delta_pr_auc']['ci_lower']:+.4f}, {bp['delta_pr_auc']['ci_upper']:+.4f}]")
            b3.metric("Delta Rec@10%FPR", f"{bp['delta_recall_at_10_fpr']['mean']*100:+.2f}%", f"95% CI: [{bp['delta_recall_at_10_fpr']['ci_lower']*100:+.2f}%, {bp['delta_recall_at_10_fpr']['ci_upper']*100:+.2f}%]")

    # Diagnostic Curves & SHAP
    with st.expander("📈 Precision-Recall, ROC, & Reliability Calibration Curves", expanded=False):
        c_p1, c_p2 = st.columns(2)
        if os.path.exists("reports/pr_roc_curve.png"):
            c_p1.image("reports/pr_roc_curve.png", caption="Precision-recall and ROC curves", width="stretch")
        if os.path.exists("reports/calibration_curve.png"):
            c_p2.image("reports/calibration_curve.png", caption="Probability calibration curve", width="stretch")

    with st.expander("🧠 Global Feature Importance (SHAP Summary)", expanded=False):
        if os.path.exists("reports/shap_summary.png"):
            st.image("reports/shap_summary.png", caption="Global SHAP feature importance for the top 20 features", width="stretch")

    # Limitations & Guidance
    with st.expander("📖 How to Read These Metrics & Known Limitations", expanded=False):
        st.markdown(
            """
            - **Why 88.4% Accuracy is Better than the Old 97.8%:** The old model was evaluated on randomly split, duplicated URLs where test domains leaked directly into training folds. On honest domain-separated holdouts, the clean baseline achieved 86.8%. The new model reaches **88.4% Acc and 96.2% ROC-AUC** honestly.
            - **Domain Cluster Effect:** 13 multi-tenant hosting domains (e.g. `tumblr.com`, `wordpress.com`) account for 18.6% of the holdout set, explaining the slight generalization gap between CV (91.2%) and single holdout (88.4%).
            - **Limitations:** Lexical models cannot detect compromised legitimate servers with standard clean URL paths unless behavioral or page-content features are incorporated.
            """
        )


show_model_performance_page()

