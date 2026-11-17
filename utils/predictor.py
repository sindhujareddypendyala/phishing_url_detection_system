import json
import os
import joblib
from urllib.parse import urlparse
from utils.feature_extraction import extract_features

# Load model pipeline
model = joblib.load("models/live_model.joblib")

# Load threshold configuration if available
DEFAULT_THRESHOLD = 0.36
BALANCED_THRESHOLD = 0.49
THRESHOLD_CONFIG_PATH = "models/threshold_config.json"
if os.path.exists(THRESHOLD_CONFIG_PATH):
    try:
        with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
            t_cfg = json.load(f)
            DEFAULT_THRESHOLD = float(t_cfg.get("default_threshold", 0.36))
            BALANCED_THRESHOLD = float(t_cfg.get("balanced_threshold", 0.49))
    except Exception:
        pass

TRUSTED_DOMAINS = [
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
]

def predict_url(url: str, threshold: float = None):
    """
    Predict phishing probability and binary label with configurable threshold and risk rating.
    Returns: (prediction: int, probability: float, features: list, risk_level: str)
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    domain = urlparse(url).netloc.lower().split(":")[0]

    for trusted in TRUSTED_DOMAINS:
        if domain == trusted or domain.endswith("." + trusted):
            return 0, 0.01, extract_features(url), "Low"

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