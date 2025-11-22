import streamlit as st
import pandas as pd
import numpy as np
import re
import socket
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Phishing URL Detector", layout="wide")
st.title("🛡 Phishing URL Detection using Machine Learning")


# -------------------------
# FEATURE EXTRACTION
# -------------------------
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
    features["contains_suspicious_words"] = 1 if any(w in url.lower() 
        for w in ["verify","update","login","secure","bank","free"]) else 0
    
    return list(features.values())


# -------------------------
# ML MODEL TRAINING
# -------------------------
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


# -------------------------
# FINAL URL VALIDITY CHECK (NO HTTP)
# -------------------------
def domain_exists(url):
    """Check if domain exists through DNS lookup."""
    try:
        domain = urlparse(url).netloc
        socket.gethostbyname(domain)
        return True
    except:
        return False


# -------------------------
# UI
# -------------------------
user_url = st.text_input("🔍 Enter URL to check:")

if user_url:

    # 1️⃣ URL FORMAT VALIDATION
    if not user_url.startswith(("http://", "https://")):
        st.error("❌ Invalid URL format")
        st.stop()

    # 2️⃣ DNS CHECK
    if not domain_exists(user_url):
        st.error("🚨 Domain does NOT exist — Phishing URL!")
        st.stop()

    # 3️⃣ ML PREDICTION
    feat = extract_features(user_url)
    pred = model.predict([feat])[0]
    prob = model.predict_proba([feat])[0][1]

    if pred == 1:
        st.error(f"🚨 Phishing URL Detected!\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Safe URL\nPhishing Probability: {prob:.2f}")