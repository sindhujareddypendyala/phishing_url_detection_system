"""
Train and serialize the final uncalibrated LightGBM production pipeline
on 100% of Train+Val, and compute out-of-fold threshold configurations.
"""

import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

from utils.feature_extraction import (
    normalize_url,
    extract_registered_domain,
    URLFeatureExtractor,
    FEATURE_NAMES
)

def main():
    print("=" * 70)
    print("TRAINING FINAL PRODUCTION PIPELINE (Uncalibrated LightGBM)")
    print("=" * 70)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Load Data
    df = pd.read_csv("data/phishingURL.csv")
    df["norm_url"] = df["url"].apply(normalize_url)
    df_clean = df.drop_duplicates(subset=["norm_url"]).copy().reset_index(drop=True)
    df_clean["registered_domain"] = df_clean["norm_url"].apply(extract_registered_domain)

    X = df_clean["norm_url"].values
    y = (df_clean["status"] == "phishing").astype(int).values
    groups = df_clean["registered_domain"].values

    # 2. Holdout split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))

    X_tr_val, y_tr_val = X[tr_idx], y[tr_idx]
    groups_tr_val = groups[tr_idx]
    print(f"[+] Loaded Train+Val: {len(X_tr_val)} URLs across {len(set(groups_tr_val))} registered domains")

    # 3. Build Feature Union Pipeline
    feature_union = FeatureUnion([
        ("lexical", Pipeline([
            ("extract", URLFeatureExtractor()),
            ("scale", StandardScaler())
        ])),
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=5,
            max_features=1000
        ))
    ])

    lgbm_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    # 4. Out-of-fold threshold search on Train+Val (5 folds)
    print("[+] Computing Out-of-Fold predictions on Train+Val for threshold optimization...")
    X_tr_val_feats = feature_union.fit_transform(X_tr_val)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_tr_val))

    for tr_i, va_i in sgkf.split(X_tr_val_feats, y_tr_val, groups=groups_tr_val):
        m = LGBMClassifier(**lgbm_params)
        m.fit(X_tr_val_feats[tr_i], y_tr_val[tr_i])
        oof_probs[va_i] = m.predict_proba(X_tr_val_feats[va_i])[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f1, best_f1_th = -1, 0.50
    high_rec_th = 0.50
    best_prec_at_95 = -1

    for th in thresholds:
        preds = (oof_probs >= th).astype(int)
        rec = recall_score(y_tr_val, preds, zero_division=0)
        prec = precision_score(y_tr_val, preds, zero_division=0)
        f1 = f1_score(y_tr_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_th = round(float(th), 2)
        if rec >= 0.95 and prec > best_prec_at_95:
            best_prec_at_95 = prec
            high_rec_th = round(float(th), 2)

    preds_best = (oof_probs >= best_f1_th).astype(int)
    preds_hr = (oof_probs >= high_rec_th).astype(int)

    print(f"    Best-F1 OOF Threshold: {best_f1_th:.2f} (F1 = {best_f1*100:.2f}%, Prec = {precision_score(y_tr_val, preds_best)*100:.2f}%, Rec = {recall_score(y_tr_val, preds_best)*100:.2f}%)")
    print(f"    High-Recall (>=95%) OOF Threshold: {high_rec_th:.2f} (Rec = {recall_score(y_tr_val, preds_hr)*100:.2f}%, Prec = {precision_score(y_tr_val, preds_hr)*100:.2f}%, F1 = {f1_score(y_tr_val, preds_hr)*100:.2f}%)")

    threshold_config = {
        "default_threshold": best_f1_th,
        "balanced_threshold": best_f1_th,
        "high_recall_threshold": high_rec_th,
        "best_f1_specs": {
            "threshold": best_f1_th,
            "oof_f1": float(best_f1),
            "oof_precision": float(precision_score(y_tr_val, preds_best)),
            "oof_recall": float(recall_score(y_tr_val, preds_best))
        },
        "high_recall_specs": {
            "threshold": high_rec_th,
            "oof_f1": float(f1_score(y_tr_val, preds_hr)),
            "oof_precision": float(precision_score(y_tr_val, preds_hr)),
            "oof_recall": float(recall_score(y_tr_val, preds_hr))
        }
    }

    with open("models/threshold_config.json", "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=2)
    print("[+] Saved 'models/threshold_config.json'")

    # 5. Fit full pipeline on 100% of Train+Val
    print("[+] Fitting final production pipeline on 100% of Train+Val...")
    final_pipeline = Pipeline([
        ("features", feature_union),
        ("classifier", LGBMClassifier(**lgbm_params))
    ])
    final_pipeline.fit(X_tr_val, y_tr_val)

    # 6. Serialize model
    prod_path = "models/live_model.joblib"
    joblib.dump(final_pipeline, prod_path)
    model_size_mb = os.path.getsize(prod_path) / (1024 * 1024)
    print(f"[+] Saved final pipeline to '{prod_path}' ({model_size_mb:.2f} MB)")
    assert model_size_mb < 50.0, f"Error: model exceeds 50 MB ({model_size_mb:.2f} MB)"

if __name__ == "__main__":
    main()
