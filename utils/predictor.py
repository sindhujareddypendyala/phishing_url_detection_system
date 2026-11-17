import joblib
from urllib.parse import urlparse
from utils.feature_extraction import (
    extract_features
)

model = joblib.load(
    "models/live_model.joblib"
)

def predict_url(url):

    

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

def predict_url(url):

    domain = urlparse(url).netloc.lower()

    for trusted in TRUSTED_DOMAINS:

        if domain == trusted or domain.endswith("." + trusted):

            return 0, 0.99, []

    features = extract_features(url)

    prediction = model.predict([features])[0]

    probability = model.predict_proba([features])[0][1]

    return prediction, probability, features