import streamlit as st
import pandas as pd
import numpy as np
import re
import socket
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib
import io
import time

# Try to import Lottie helper
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

# -------------------------
# App config + CSS (Dark Mode D: Grey Soft Dark)
# -------------------------
st.set_page_config(page_title="🕵️‍♀️Phishing URL Detection System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark theme + animations + cards
st.markdown(
    """
    <style>
    :root{
        --bg:#111217;
        --card:#18191b;
        --muted:#9aa0a6;
        --accent:#7f8c8d;
        --success:#16a34a;
        --danger:#ef4444;
    }

    /* page background */
    .stApp {
    background: linear-gradient(180deg, #2b2d31 0%, #1f2124 100%);
    color: navyblue;
    }



    /* container card */
    .card {
        background: var(--card);
        padding: 28px;
        border-radius: 15px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.5);
        margin-bottom: 12px;
    }

    .big-card {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        box-shadow: 0 8px 30px rgba(0,0,0,0.6);
    }

    .title {
        font-size: 30px;
        font-weight: 700;
        color: #f1f5f9;
    }

    .muted { color: var(--muted); font-size: 13px; }

    /* nice buttons */
    .nav-btn {
        background: transparent;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 10px 16px;
        border-radius: 10px;
    }

    /* Lottie container sizing */
    .lottie {
        max-width: 220px;
        margin: auto;
    }

    /* Toast-like notification */
    .toast {
        padding: 14px 16px;
        border-radius: 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.04);
        margin: 8px 0;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeIn 0.6s ease-out both; }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    .pulse { animation: pulse 1.6s ease-in-out infinite; }

    /* big prediction number */
    .pred-big {
        font-size: 36px; font-weight: 800; letter-spacing: -0.5px;
    }

    /* small helper */
    .small { font-size: 13px; color: var(--muted); }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# LOTTIE URL placeholders (You can replace these with your favorite Lottie URLs or local JSON)
# -------------------------
LOTTIE_URLS = {
    "shield": "https://assets1.lottiefiles.com/packages/lf20_u4yrau.json",        # placeholder: cyber security shield
    "warning": "https://assets2.lottiefiles.com/packages/lf20_jtbfg2nb.json",    # placeholder: phishing warning
    "check": "https://assets3.lottiefiles.com/packages/lf20_totrp3xk.json",      # placeholder: secure verified
    "loading": "https://assets4.lottiefiles.com/packages/lf20_cg8q6x2v.json",    # placeholder: loading
    "scanner": "https://assets5.lottiefiles.com/packages/lf20_6dr6wery.json"     # placeholder: url scanner
}

# If streamlit_lottie not available, we will show static text and instructions
if not LOTTIE_AVAILABLE:
    st.sidebar.warning("Install `streamlit_lottie` for Lottie animations: `pip install streamlit_lottie`")

# -------------------------
# Feature extraction (improved)
# -------------------------
def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path + (("?" + parsed.query) if parsed.query else "")

    features = {}
    features["url_length"] = len(url)
    features["has_https"] = 1 if url.lower().startswith("https") else 0
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_slashes"] = url.count("/")
    features["num_at"] = url.count("@")
    features["num_question"] = url.count("?")
    features["num_equals"] = url.count("=")
    features["has_ip"] = 1 if re.match(r"^\d+\.\d+\.\d+\.\d+$", hostname) else 0
    features["domain_length"] = len(hostname)
    features["path_length"] = len(path)
    features["contains_https_token"] = 1 if "https" in hostname.lower() and not url.startswith("https") else 0
    features["contains_suspicious_words"] = 1 if any(w in url.lower() for w in [
        "verify", "update", "login", "secure", "bank", "free", "account", "confirm", "signin", "admin"
    ]) else 0

    # return as ordered list
    ordered = [
        features["url_length"], features["has_https"], features["num_dots"], features["num_hyphens"],
        features["num_slashes"], features["num_at"], features["num_question"], features["num_equals"],
        features["has_ip"], features["domain_length"], features["path_length"],
        features["contains_https_token"], features["contains_suspicious_words"]
    ]
    return ordered

FEATURE_COLUMNS = [
    "url_length","has_https","num_dots","num_hyphens","num_slashes","num_at",
    "num_question","num_equals","has_ip","domain_length","path_length",
    "contains_https_token","contains_suspicious_words"
]

# -------------------------
# Model: RandomForest (Option A)
# - If a pre-saved model exists in working dir ('rf_model.joblib'), it will be loaded.
# - Otherwise the app will synthesize+train a realistic-ish dataset on first run (small but robust).
# -------------------------
MODEL_PATH = "rf_model.joblib"

@st.cache_resource
def get_model():
    # Try load
    try:
        clf = joblib.load(MODEL_PATH)
        return clf, True
    except Exception:
        # Build synthetic training set using heuristics + some randomization
        np.random.seed(42)
        rows = 2000  # synthetic size (quick to train)
        data = []
        labels = []
        suspicious_tokens = ["verify", "update", "login", "secure", "bank", "free", "confirm", "signin", "admin"]

        for _ in range(rows):
            is_phish = np.random.rand() < 0.25  # ~25% phishing
            # build synthetic url
            domain_len = np.random.randint(5, 35) if not is_phish else np.random.randint(10, 60)
            url_len = np.random.randint(20, 60) if not is_phish else np.random.randint(40, 180)
            num_dots = np.random.randint(1, 4) if not is_phish else np.random.randint(2, 8)
            num_hyphens = np.random.poisson(0.3) if not is_phish else np.random.poisson(2.0)
            num_slash = np.random.randint(1, 5) if not is_phish else np.random.randint(2, 12)
            num_at = 0 if not is_phish else (np.random.choice([0,1], p=[0.8,0.2]))
            num_question = 0 if not is_phish else np.random.choice([0,1,2], p=[0.7,0.2,0.1])
            num_equals = 0 if not is_phish else np.random.choice([0,1,2], p=[0.7,0.2,0.1])
            has_ip = 1 if (is_phish and np.random.rand() < 0.08) else 0
            contains_https_token = 1 if (is_phish and np.random.rand() < 0.12) else 0
            contains_suspicious = 1 if (is_phish and np.random.rand() < 0.6) else 0

            row = [
                url_len, int(np.random.rand() < 0.9 if not is_phish else 0.2),
                num_dots, num_hyphens, num_slash, num_at, num_question, num_equals, has_ip,
                domain_len, max(1, url_len - domain_len - 10),
                contains_https_token, contains_suspicious
            ]
            data.append(row)
            labels.append(1 if is_phish else 0)

        X = np.array(data)
        y = np.array(labels)
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        # Save model
        try:
            joblib.dump(clf, MODEL_PATH)
        except Exception:
            pass
        return clf, False

model, loaded_from_file = get_model()

# -------------------------
# Utility: domain exists check
# -------------------------
def domain_exists(url, timeout=2.0):
    try:
        domain = urlparse(url).netloc
        if not domain:
            return False
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(domain)
        return True
    except Exception:
        return False

# -------------------------
# UI Layout
# -------------------------
st.markdown("<div class='card fade-in'><div class='title'>🛡 Phishing URL Detection System</div>"
            "<div class='muted'>Dark mode • RandomForest model • Lottie animations • Bulk CSV</div></div>", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.subheader("Home")
    st.write("Welcome! Use the buttons to navigate")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if LOTTIE_AVAILABLE:
        # show a couple of Lotties (shield + loading)
        try:
            st_lottie = st_lottie  # silence lint
            with st.container():
                st.markdown("<div class='lottie fade-in'>", unsafe_allow_html=True)
                st_lottie_url = LOTTIE_URLS.get("shield")
                try:
                    st_lottie(st_lottie_url, height=140)
                except Exception:
                    st.write("🔔 Lottie failed to load — check your URL or internet.")
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            pass
    else:
        st.markdown("Lottie not available. Install `streamlit_lottie` to see animations.")
    st.markdown("</div>", unsafe_allow_html=True)

# Navigation (Home / Single / Bulk / Train)
page = st.sidebar.selectbox("Navigate", ["Start", "Single URL", "Bulk CSV", "Train Model / Diagnostics", "About"])

# ---------- HOME / START ----------
if page == "Start":
    st.markdown("<div class='big-card fade-in'>", unsafe_allow_html=True)
    st.title("Welcome👋")
    st.write("Press the big button to start URL inspection.")
    # Lottie + welcome button
    if LOTTIE_AVAILABLE:
        try:
            st_lottie(LOTTIE_URLS["scanner"], height=200)
        except Exception:
            pass
    if st.button("Start Detection 🔍", key="start_button", help="Go to Single URL mode"):
        # go to single - emulate by setting page var (frontend only)
        page = "Single URL"
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- SINGLE URL ----------
if page == "Single URL":
    st.markdown("<div class='card fade-in'><h3>🔎 Single URL Checker</h3></div>", unsafe_allow_html=True)
    user_url = st.text_input("Enter full URL ", placeholder=" ")
 
    # small hint + lottie
    st.markdown("<div class='small muted'>Tip: you can paste a URL or upload CSV from the sidebar 'Bulk CSV'.</div>", unsafe_allow_html=True)

    if user_url:
        # format check
        if not user_url.lower().startswith(("http://", "https://")):
            st.markdown("<div class='toast'>❌ Invalid URL format. Make sure to include http:// or https://</div>", unsafe_allow_html=True)
        else:
            # DNS check (fast)
            dns_ok = domain_exists(user_url)
            if not dns_ok:
                st.markdown("<div class='toast' style='border-left:4px solid #ef4444'>🚨 Domain does NOT resolve — likely phishing</div>", unsafe_allow_html=True)

            # extract and predict
            feat = extract_features(user_url)
            pred = model.predict([feat])[0]
            prob = model.predict_proba([feat])[0][1]

            # Show big prediction card
            color = "#ef4444" if pred == 1 else "#16a34a"
            label = "Phishing" if pred == 1 else "Safe"
            confidence_pct = f"{prob*100:.1f}%"

            st.markdown(f"""
                <div class='big-card fade-in' style='border:1px solid rgba(255,255,255,0.03)'>
                    <div style='display:flex; align-items:center; justify-content:space-between; gap:20px'>
                        <div>
                            <div class='muted'>Result</div>
                            <div class='pred-big' style='color:{color};'>{label}</div>
                            <div class='small muted'>Confidence: <strong>{confidence_pct}</strong></div>
                            <div style='margin-top:8px' class='small muted'>DNS Resolved: <strong>{dns_ok}</strong></div>
                        </div>
                        <div style='text-align:center;'>
            """, unsafe_allow_html=True)

            # lottie thumb
            if LOTTIE_AVAILABLE:
                try:
                    if pred == 1:
                        st_lottie(LOTTIE_URLS["warning"], height=140)
                    else:
                        st_lottie(LOTTIE_URLS["check"], height=140)
                except Exception:
                    pass
            st.markdown("</div></div></div>", unsafe_allow_html=True)

            # Notification padding (toast-like)
            
            # show extracted features for transparency
            st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
            st.subheader("Features (for explainability)")
            feat_df = pd.DataFrame([feat], columns=FEATURE_COLUMNS)
            st.dataframe(feat_df.T.rename_axis("feature").rename(columns={0:"value"}), height=260)
            st.markdown("</div>", unsafe_allow_html=True)

# ---------- BULK CSV ----------
if page == "Bulk CSV":
    st.markdown("<div class='card fade-in'><h3>📁 Bulk CSV Prediction</h3></div>", unsafe_allow_html=True)
    st.write("Upload a CSV with a column named `url`. The app will predict and allow download of results.")
    uploaded = st.file_uploader("Upload CSV file (must contain 'url' column)", type=["csv"])

    if uploaded:
        df_in = pd.read_csv(uploaded)
        if "url" not in df_in.columns:
            st.error("CSV must contain a 'url' column ")
        else:
            st.info(f"Loaded {len(df_in)} rows. Running predictions...")
            # progress bar
            progress = st.progress(0)
            results = []
            probs = []
            dns_flags = []
            for i, u in enumerate(df_in["url"].astype(str).tolist()):
                dns_ok = domain_exists(u)
                dns_flags.append(dns_ok)
                if not u.lower().startswith(("http://","https://")):
                    # treat invalid-format as phishing
                    pred = 1
                    prob = 0.98
                else:
                    feat = extract_features(u)
                    pred = int(model.predict([feat])[0])
                    prob = float(model.predict_proba([feat])[0][1])
                results.append("Phishing" if pred == 1 else "Safe")
                probs.append(prob)
                # update progress
                if i % 10 == 0:
                    progress.progress(min(100, int((i/len(df_in))*100)))
            progress.progress(100)
            df_out = df_in.copy()
            df_out["dns_resolved"] = dns_flags
            df_out["prediction"] = results
            df_out["phishing_prob"] = probs

            st.success("Bulk prediction completed ✔️")
            # show top rows
            st.dataframe(df_out.head(10))

            # Visualizations
            st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
            st.subheader("Prediction Distribution")
            fig, ax = plt.subplots(figsize=(6,3))
            df_out["prediction"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, wedgeprops={'alpha':0.9}, ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

            # Download button
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", csv_bytes, file_name="phishing_results.csv", mime="text/csv")

# ---------- TRAIN / DIAGNOSTICS ----------
if page == "Train Model / Diagnostics":
    st.markdown("<div class='card fade-in'><h3>🧪 Train model or evaluate diagnostics</h3></div>", unsafe_allow_html=True)
    st.write("You can retrain the RandomForest here with a larger synthetic dataset or upload a labeled CSV (with columns `url` and `label`) to train a custom model.")

    train_choice = st.radio("Choose training input:", ["Use synthetic (fast)", "Upload labeled CSV"])
    if train_choice == "Upload labeled CSV":
        labeled_file = st.file_uploader("Upload CSV with 'url' and 'label' columns", type=["csv"])
        if labeled_file:
            df_lab = pd.read_csv(labeled_file)
            if not {"url","label"}.issubset(df_lab.columns):
                st.error("CSV must contain 'url' and 'label' columns.")
            else:
                st.info(f"Training on {len(df_lab)} rows...")
                # build features
                X = []
                y = []
                for _, r in df_lab.iterrows():
                    X.append(extract_features(str(r["url"])))
                    y.append(int(r["label"]))
                X = np.array(X); y = np.array(y)
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
                clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
                with st.spinner("Training... this may take a bit"):
                    clf.fit(X_train, y_train)
                    preds = clf.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    st.success(f"Trained. Test accuracy: {acc:.3f}")
                    st.text("Classification report:")
                    st.text(classification_report(y_test, preds))
                    # Save model
                    joblib.dump(clf, MODEL_PATH)
                    st.success(f"Saved model to {MODEL_PATH}")
    else:
        # synthetic retrain button
        if st.button("Retrain on synthetic larger dataset (quick)"):
            st.info("Retraining...")
            # reuse get_model trick by deleting cache and calling get_model again
            # (streamlit.cache_resource can't be cleared easily here; we train inline)
            np.random.seed(0)
            rows = 5000
            data = []
            labels = []
            for _ in range(rows):
                is_phish = np.random.rand() < 0.25
                domain_len = np.random.randint(5, 35) if not is_phish else np.random.randint(10, 60)
                url_len = np.random.randint(20, 60) if not is_phish else np.random.randint(40, 180)
                num_dots = np.random.randint(1, 4) if not is_phish else np.random.randint(2, 8)
                num_hyphens = np.random.poisson(0.3) if not is_phish else np.random.poisson(2.0)
                num_slash = np.random.randint(1, 5) if not is_phish else np.random.randint(2, 12)
                num_at = 0 if not is_phish else (np.random.choice([0,1], p=[0.8,0.2]))
                num_question = 0 if not is_phish else np.random.choice([0,1,2], p=[0.7,0.2,0.1])
                num_equals = 0 if not is_phish else np.random.choice([0,1,2], p=[0.7,0.2,0.1])
                has_ip = 1 if (is_phish and np.random.rand() < 0.08) else 0
                contains_https_token = 1 if (is_phish and np.random.rand() < 0.12) else 0
                contains_suspicious = 1 if (is_phish and np.random.rand() < 0.6) else 0
                row = [
                    url_len, int(np.random.rand() < 0.9 if not is_phish else 0.2),
                    num_dots, num_hyphens, num_slash, num_at, num_question, num_equals, has_ip,
                    domain_len, max(1, url_len - domain_len - 10),
                    contains_https_token, contains_suspicious
                ]
                data.append(row); labels.append(1 if is_phish else 0)
            X = np.array(data); y = np.array(labels)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42)
            with st.spinner("Training..."):
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds)
                joblib.dump(clf, MODEL_PATH)
                st.success(f"Retrained and saved model (accuracy: {acc:.2f})")

# ---------- ABOUT ----------
if page == "About":
    st.markdown("<div class='card fade-in'><h3>About this App</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    - Model: **RandomForest (Option A)** trained on a synthetic but realistic feature distribution. 
    - Dark theme: Grey Soft Dark (professional).
    - Animations: `fade-in` on cards and `pulse` for subtle liveliness.
    - Lottie: 5 placeholder Lottie animations are included (replace URLs in `LOTTIE_URLS` if you prefer).
    - Bulk CSV: Upload `url` column; results include `prediction`, `phishing_prob`, and `dns_resolved`.
    """)
    st.markdown("**Notes & next steps:**")
    st.markdown("""
    1. To get production-grade accuracy, train with a large labeled dataset (Phishing + Legitimate URLs) and perform feature engineering (domain age, WHOIS, TLS cert checks, IP geolocation, content-based signals).  
    2. You can replace Lottie URLs with your favorite JSONs from LottieFiles.  
    3. Add real-time blacklist checks (PhishTank, Google Safe Browsing) for better safety.
    """)
    if LOTTIE_AVAILABLE:
        try:
            st_lottie(LOTTIE_URLS["scanner"], height=200)
        except Exception:
            pass

# -------------------------
# Footer / small animation hint
# -------------------------
st.markdown("<div style='margin-top:12px' class='muted small'>Built for demo & learning • Customize Lottie URLs and labeled datasets to improve performance</div>", unsafe_allow_html=True)
