import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import plotly.express as px

# -------------------------
# PAGE
# -------------------------

def show_analytics():

    st.markdown(
        """
        <div class='main-title'>
        📊 Analytics Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # -------------------------
    # LOAD DATA
    # -------------------------

    try:

        df = pd.read_csv(
            "data/phishingURL.csv"
        )

    except Exception as e:

        st.error(
            f"Dataset Error: {e}"
        )

        return

    # -------------------------
    # PREPARE DATA
    # -------------------------

    X = df.drop(
        ["url", "status"],
        axis=1
    )

    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

    model = joblib.load(
        "models/phishing_model.joblib"
    )

    y_pred = model.predict(X_test)

    # -------------------------
    # METRICS
    # -------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="phishing"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="phishing"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="phishing"
    )

    # -------------------------
    # METRIC CARDS
    # -------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision*100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall*100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1*100:.2f}%"
    )

    st.write("")

    # -------------------------
    # DATASET INFO
    # -------------------------

    st.subheader(
        "📂 Dataset Overview"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total URLs",
        len(df)
    )

    c2.metric(
        "Features",
        X.shape[1]
    )

    c3.metric(
        "Classes",
        y.nunique()
    )

    # -------------------------
    # CLASS DISTRIBUTION
    # -------------------------

    st.subheader(
        "📈 Class Distribution"
    )

    class_df = (
        y.value_counts()
        .reset_index()
    )

    class_df.columns = [
        "Class",
        "Count"
    ]

    fig = px.pie(

        class_df,

        values="Count",

        names="Class",

        hole=0.4,

        title="Legitimate vs Phishing URLs"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------
    # CONFUSION MATRIX
    # -------------------------

    st.subheader(
        "🎯 Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Legitimate", "Phishing"],
        columns=["Legitimate", "Phishing"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------

    if hasattr(
        model,
        "feature_importances_"
    ):

        st.subheader(
            "🔥 Top 20 Features"
        )

        importance_df = pd.DataFrame({

            "Feature":
            X.columns,

            "Importance":
            model.feature_importances_
        })

        importance_df = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(20)
        )

        fig = px.bar(

            importance_df,

            x="Importance",

            y="Feature",

            orientation="h",

            title=
            "Most Important Features"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------
    # DATA PREVIEW
    # -------------------------

    st.subheader(
        "📝 Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )