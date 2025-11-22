import streamlit as st
import pandas as pd
import numpy as np
import re
import socket
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from streamlit_lottie import st_lottie
import json
import base64

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------
# DARK MODE CUSTOM CSS
# -----------------------------------------------------------
st.markdown("""
<style>

* { font-family: 'Segoe UI', sans-serif; }

[data-testid="stAppViewContainer"] {
    background-color: #1e1e1e;
    color: #f5f5f5;
}

/* Big animated card */
.big-card {
    padding: 30px;
    background: #2a2a2a;
    border-radius: 20px;
    border: 1px solid #444;
    animation: fadeIn 1s ease-in-out;
}

/* Fade-in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Pulse animation */
.pulse {
    animation: pulseAnim 2s infinite;
}
@keyframes pulseAnim {
    0% { transform: scale(1); }
    50% { transform: scale(1.03); }
    100% { transform: scale(1); }
}

.sidebar-text {
    color: white;
}

/* Buttons */
.stButton>button {
    background-color: #3a3a3a;
    color: white;
    padding: 10px 20px;
    border-radius: 12px;
    border: 1px solid #555;
}
.stButton>button:hover {
    background-color: #505050;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# LOAD LOTTIE FILES
# -----------------------------------------------------------
def load_lottie(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

lottie_secure = load_lottie("lottie/secure.json")      # 1
lottie_phishing = load_lottie("lottie/phishing.json")  # 2
lottie_verified = load_lottie("lottie/verified.json")  # 3
lottie_loading = load_lottie("lottie/loading.json")    # 4
lottie_scan = load_lottie("lottie/scan.json")          # 5


# -----------------------------------------------------------
# FEATURE EXTRACTION FUNCTION
# -----------------------------------------------------------
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
        for w in ["verify", "update", "login", "secure", "bank", "free"]) else 0
    
    return list(features.values())


# -----------------------------------------------------------
# TRAIN A REAL MODEL (Random Forest)
# -----------------------------------------------------------
data = pd.read_csv("dataset.csv")   # <--- Replace with your real phishing dataset
X = data.drop("label", axis=1)
y = data["label"]

model = RandomForestClassifier()
model.fit(X, y)


# -----------------------------------------------------------
# DNS CHECK
# -----------------------------------------------------------
def domain_exists(url):
    try:
        domain = urlparse(url).netloc
        socket.gethostbyname(domain)
        return True
    except:
        return False


# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🔎 Single URL Checker", "📁 Bulk CSV Scanner"])

st.sidebar.markdown("---")
st.sidebar.subheader("Bulk CSV Upload")
st.sidebar.text("Upload CSV containing URLs:")

uploaded_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_csv:
    st.sidebar.success("CSV uploaded!")

st.sidebar.markdown("---")
st.sidebar.caption("Dark Mode • Lottie • Animations Enabled")


# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if page == "🏠 Home":
    col1, col2 = st.columns([1, 1])
    with col1:
        st_lottie(lottie_secure, height=300)
    with col2:
        st.title("🛡 Phishing URL Detection System")
        st.markdown("""
        ### Powered by Machine Learning  
        - Real Random Forest Model  
        - DNS Validation  
        - URL Structure Analysis  
        - Suspicious Keyword Detection  
        - Bulk CSV Scanner  
        """)

    st.button("🔎 Start Checking URLs", key="start_btn")


# -----------------------------------------------------------
# SINGLE URL CHECKER
# -----------------------------------------------------------
elif page == "🔎 Single URL Checker":

    st.title("🔎 Single URL Checker")
    user_url = st.text_input("Enter full URL (include http:// or https://):")

    if user_url:

        # Format check
        if not user_url.startswith(("http://", "https://")):
            st.error("❌ Invalid URL format.")
            st.stop()

        # DNS check
        dns_ok = domain_exists(user_url)

        if not dns_ok:
            st_lottie(lottie_phishing, height=200)
            st.error("🚨 Domain does NOT resolve — likely phishing")
            st.stop()

        # ML prediction
        feat = extract_features(user_url)
        pred = model.predict([feat])[0]
        prob = model.predict_proba([feat])[0][1]

        # Final decision card
        st.markdown('<div class="big-card pulse">', unsafe_allow_html=True)

        if pred == 1:
            st_lottie(lottie_phishing, height=160)
            st.error(f"🚨 **PHISHING DETECTED!**\nConfidence: {prob*100:.2f}%")
        else:
            st_lottie(lottie_verified, height=160)
            st.success(f"✅ URL appears SAFE\nPhishing Probability: {prob*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------
# BULK CSV SCANNER
# -----------------------------------------------------------
elif page == "📁 Bulk CSV Scanner":

    st.title("📁 Bulk CSV URL Scanner")

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        if "url" not in df.columns:
            st.error("CSV must contain a column named 'url'.")
            st.stop()

        st_lottie(lottie_loading, height=150)

        results = []

        for url in df["url"]:
            row = {"url": url}

            dns_ok = domain_exists(url)
            row["dns_resolved"] = dns_ok

            if not dns_ok:
                row["prediction"] = "Phishing"
                row["confidence"] = 1.0
            else:
                feat = extract_features(url)
                p = model.predict([feat])[0]
                c = model.predict_proba([feat])[0][1]
                row["prediction"] = "Phishing" if p == 1 else "Safe"
                row["confidence"] = float(c)

            results.append(row)

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Download button
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Results CSV", csv, "scan_results.csv", "text/csv")

    else:
        st.info("Upload a CSV file from the left sidebar to begin scanning.")
