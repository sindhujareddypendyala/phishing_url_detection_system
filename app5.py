import streamlit as st
import pandas as pd
import numpy as np
import socket
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
import joblib
import re

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Phishing URL Detector", layout="wide")

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
        /* Light Black Background */
        [data-testid="stAppViewContainer"] {
            background-color: #1f1f1f;
        }
        [data-testid="stSidebar"] {
            background-color: #2a2a2a;
        }
        .title {
            color: white;
            text-align: center;
            font-size: 40px;
            font-weight: 700;
            padding-bottom: 10px;
        }
        .input-box input {
            background-color: #333 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.markdown('<div class="title">🔍 Phishing URL Detection</div>', unsafe_allow_html=True)

# Load ML Model
model = joblib.load("phish_model.joblib")

# ---------------- Feature Extraction ----------------
def extract_features(url):
    parsed = urlparse(url)
    features = {
        "url_length": len(url),
        "hostname_length": len(parsed.netloc),
        "path_length": len(parsed.path),
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "num_at": url.count('@'),
        "num_slashes": url.count('/'),
        "num_digits": sum(c.isdigit() for c in url),
        "https": 1 if url.startswith("https") else 0
    }
    return pd.DataFrame([features])

# ---------------- DNS Check ----------------
def dns_exists(url):
    try:
        domain = urlparse(url).netloc
        socket.gethostbyname(domain)
        return True
    except:
        return False

# ---------------- URL Input ----------------
st.markdown("<h4 style='color:white;'>Enter URL:</h4>", unsafe_allow_html=True)
url = st.text_input("", placeholder="https://example.com", label_visibility="collapsed")

if url:
    # Feature Extraction
    features = extract_features(url)

    # Prediction (1 = phishing, 0 = safe)
    prediction = model.predict(features)[0]

    # ---------------- Alert Card ----------------
    if prediction == 1:
        message = "⚠️ WARNING: This URL is PHISHING. Do NOT use it!"
        card_color = "#b91c1c"  # red
    else:
        message = "🟢 This URL is SAFE to use."
        card_color = "#15803d"  # green

    st.markdown(
        f"""
        <div style="
            background:{card_color};
            padding:35px;
            border-radius:18px;
            text-align:center;
            color:white;
            font-size:32px;
            font-weight:700;
            box-shadow:0 4px 15px rgba(0,0,0,0.3);
            margin-top:20px;
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )
