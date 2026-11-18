import json
import os
import joblib
import streamlit as st
from urllib.parse import urlparse
from utils.feature_extraction import extract_features

@st.cache_resource
def load_model():
    """Load the unchanged production pipeline once per Streamlit process."""
    return joblib.load("models/live_model.joblib")


model = load_model()

# Load threshold configuration
DEFAULT_THRESHOLD = 0.52
BALANCED_THRESHOLD = 0.52
HIGH_RECALL_THRESHOLD = 0.32

THRESHOLD_CONFIG_PATH = "models/threshold_config.json"
if os.path.exists(THRESHOLD_CONFIG_PATH):
    try:
        with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
            t_cfg = json.load(f)
            DEFAULT_THRESHOLD = float(t_cfg.get("default_threshold", 0.52))
            BALANCED_THRESHOLD = float(t_cfg.get("balanced_threshold", 0.52))
            HIGH_RECALL_THRESHOLD = float(t_cfg.get("high_recall_threshold", 0.32))
    except Exception:
        pass

TRUSTED_DOMAINS = frozenset([
    "google.com",
    "youtube.com",
    "linkedin.com",
    "github.com",
    "microsoft.com",
    "apple.com",
    "wikipedia.org",
    "python.org",
    "openai.com",
    "oracle.com",
    "facebook.com"
])

def is_trusted_allowlist(url: str) -> bool:
    """Check if the URL belongs to a known trusted platform domain (UI note only)."""
    try:
        domain = urlparse(url).netloc.lower().split(":")[0]
        for trusted in TRUSTED_DOMAINS:
            if domain == trusted or domain.endswith("." + trusted):
                return True
    except Exception:
        pass
    return False

def predict_url(url: str, threshold: float = None):
    """
    Predict phishing probability and binary label with configurable threshold and risk rating.
    Uses pure ML probability without allowlist overrides.
    Returns: (prediction: int, probability: float, features: list, risk_level: str)
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    features = extract_features(url)
    probability = float(model.predict_proba([url])[0][1])
    prediction = 1 if probability >= threshold else 0

    if probability >= 0.70:
        risk_level = "High"
    elif probability >= threshold:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return prediction, probability, features, risk_level
