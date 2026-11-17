import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset

df = pd.read_csv(
    "data/phishingURL.csv"
)

# -------------------------
# SELECT FEATURES
# -------------------------

selected_features = [

    "length_url",
    "length_hostname",
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_eq",
    "nb_slash",
    "https_token",
    "ratio_digits_url",
    "ratio_digits_host",
    "nb_subdomains",
    "prefix_suffix",
    "random_domain",
    "shortening_service",
    "phish_hints",
    "domain_age"
]

X = df[selected_features]

# Convert labels

y = df["status"].map({
    "legitimate": 0,
    "phishing": 1
})

# -------------------------
# TRAIN TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# TRAIN
# -------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# -------------------------
# EVALUATE
# -------------------------

pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    pred
)

print(
    f"Accuracy: {acc:.4f}"
)

# -------------------------
# SAVE
# -------------------------

joblib.dump(
    model,
    "models/live_model.joblib"
)

print(
    "Live Model Saved"
)