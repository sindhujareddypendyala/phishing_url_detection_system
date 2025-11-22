import streamlit as st
import pandas as pd
import numpy as np
import re
import socket
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import base64

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Phishing URL Detector", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .main-title {
            font-size: 42px; 
            text-align: center; 
            font-weight: bold; 
            color: #4B0082;
        }
        .big-card {
            padding: 30px; 
            border-radius: 20px; 
            background: #ffffffcc; 
            box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
            margin-top: 30px;
        }
        .welcome-box {
            text-align: center;
            padding: 50px;
            margin-top: 50px;
        }
        .nav-btn {
            width: 200px;
            font-size: 20px;
            margin-top: 20px;
        }
        .notif {
            padding: 12px;
        }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------
def extract_features(url):
    features = {}
    parsed = urlparse(url)

    features["url_length"] = len(url)
    features["has_https"] = 1 if url.startswith("https") else 0
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_slashes"] = url.count("/")
    features["has_ip"] = 1 if re.match(r"^\d+\.\d+\.\d+\.\d+$", parsed.netloc) else 0
    features["domain_length"] = len(parsed.netloc)
    features["contains_suspicious_words"] = 1 if any(
        w in url.lower() for w in ["verify", "update", "login", "secure", "bank", "free"]
    ) else 0

    return list(features.values())


# ---------------------------------------------------------
# ML Training (Dummy Example Data)
# ---------------------------------------------------------
data = {
    "url_length": [30, 80, 12, 55, 100, 22],
    "has_https": [1, 0, 1, 0, 0, 1],
    "num_dots": [2, 5, 1, 3, 7, 1],
    "num_hyphens": [0, 4, 0, 2, 5, 0],
    "num_slashes": [3, 8, 1, 5, 10, 2],
    "has_ip": [0, 1, 0, 0, 1, 0],
    "domain_length": [12, 30, 8, 20, 40, 10],
    "contains_suspicious_words": [0, 1, 0, 1, 1, 0],
    "label": [0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)
X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier()
model.fit(X, y)


# ---------------------------------------------------------
# Domain Exists Check
# ---------------------------------------------------------
def domain_exists(url):
    try:
        domain = urlparse(url).netloc
        socket.gethostbyname(domain)
        return True
    except:
        return False


# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------
menu = st.sidebar.radio("Navigation", ["Home", "URL Detection", "Bulk CSV Prediction"])

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
if menu == "Home":
    st.markdown("<h1 class='main-title'>🛡 Welcome to Phishing URL Detector</h1>", unsafe_allow_html=True)

    st.markdown("""
        <div class='welcome-box'>
            <h3>Detect phishing, fraud, and unsafe URLs with Machine Learning</h3>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Start Detection 🔍", key="start", use_container_width=True):
        st.switch_page("URL Detection")


# ---------------------------------------------------------
# URL DETECTION PAGE
# ---------------------------------------------------------
elif menu == "URL Detection":
    st.header("🔍 Single URL Checker")

    if st.button("⬅ Back to Home"):
        st.switch_page("Home")

    user_url = st.text_input("Enter URL to check:")

    if user_url:
        if not user_url.startswith(("http://", "https://")):
            st.error("❌ Invalid URL format", icon="🚨")
            st.markdown("<div class='notif'></div>", unsafe_allow_html=True)
            st.stop()

        if not domain_exists(user_url):
            st.error("🚨 Domain does NOT exist — Phishing URL!")
            st.markdown("<div class='notif'></div>", unsafe_allow_html=True)
            st.stop()

        feat = extract_features(user_url)
        pred = model.predict([feat])[0]
        prob = model.predict_proba([feat])[0][1]

        st.markdown("<div class='big-card'>", unsafe_allow_html=True)

        if pred == 1:
            st.error(f"🚨 **Phishing URL Detected!**\n\n### Probability: `{prob:.2f}`")
        else:
            st.success(f"✅ **Safe URL**\n\n### Phishing Probability: `{prob:.2f}`")

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# BULK CSV PREDICTION PAGE
# ---------------------------------------------------------
elif menu == "Bulk CSV Prediction":
    st.header("📁 Bulk CSV URL Checker")

    uploaded = st.file_uploader("Upload CSV with a column named 'url'")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())

        predictions = []
        for url in df["url"]:
            if domain_exists(url):
                feat = extract_features(url)
                pred = model.predict([feat])[0]
            else:
                pred = 1  # domain invalid → phishing
            predictions.append(pred)

        df["prediction"] = ["Safe" if p == 0 else "Phishing" for p in predictions]

        st.success("Prediction Completed!")

        # Visualization
        st.subheader("📊 Prediction Chart")
        fig, ax = plt.subplots()
        df["prediction"].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

        # Download Button
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Results CSV", csv, "phishing_results.csv")


