"""
Model Training and Evaluation Pipeline for Phishing URL Detection.

Features:
- Single source of truth feature extraction (Tier 1: 40 Lexical + Character 3-5 gram TF-IDF)
- Full URL normalization and deduplication
- Registered domain grouping (tldextract offline)
- GroupShuffleSplit (80/20) for holdout test set with ZERO domain overlap
- StratifiedGroupKFold (5 folds) for cross-validation
- Full sklearn Pipeline encapsulation
- Saves comparison table to reports/metrics.json and reports/metrics.md
- Saves trained production pipeline to models/live_model.joblib (< 50MB)
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, auc
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from utils.feature_extraction import (
    normalize_url,
    extract_registered_domain,
    URLFeatureExtractor,
    FEATURE_NAMES
)

RANDOM_SEED = 42

def run_pipeline():
    print("=" * 70)
    print("PHISHING URL DETECTION: MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)
    
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load Data
    data_path = "data/phishingURL.csv"
    print(f"\n[1/5] Loading raw dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"      Raw rows loaded: {len(df)}")
    
    # 2. Normalize and Deduplicate
    print("\n[2/5] Normalizing URLs and deduplicating...")
    df["norm_url"] = df["url"].apply(normalize_url)
    df_clean = df.drop_duplicates(subset=["norm_url"]).copy().reset_index(drop=True)
    dup_count = len(df) - len(df_clean)
    print(f"      Duplicates removed: {dup_count}")
    print(f"      Clean unique URLs: {len(df_clean)}")
    
    # 3. Domain Grouping
    print("      Extracting registered domains with offline tldextract...")
    df_clean["registered_domain"] = df_clean["norm_url"].apply(extract_registered_domain)
    num_domains = df_clean["registered_domain"].nunique()
    print(f"      Unique registered domains: {num_domains}")
    
    X = df_clean["norm_url"].values
    y = (df_clean["status"] == "phishing").astype(int).values
    groups = df_clean["registered_domain"].values
    
    # 4. Holdout Split: GroupShuffleSplit (80% train+val, 20% holdout test)
    print("\n[3/5] Partitioning holdout test set with GroupShuffleSplit (test_size=0.2)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]
    groups_train_val, groups_test = groups[train_val_idx], groups[test_idx]
    
    domain_leak = set(groups_train_val).intersection(set(groups_test))
    print(f"      Train+Val set: {len(X_train_val)} URLs across {len(set(groups_train_val))} domains")
    print(f"      Holdout Test set: {len(X_test)} URLs across {len(set(groups_test))} domains")
    print(f"      Domain leakage check: {len(domain_leak)} overlapping domains")
    assert len(domain_leak) == 0, "Critical Error: Domain leakage detected across split!"
    
    # 5. Define Feature Union Pipeline (Tier 1: Lexical + TF-IDF)
    print("\n[4/5] Building feature union (40 Lexical Features + Char 3-5 gram TF-IDF)...")
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
    
    # Model candidates
    model_candidates = {
        "Random Forest (Baseline)": RandomForestClassifier(
            n_estimators=150, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_SEED
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=RANDOM_SEED, n_jobs=-1, eval_metric="logloss"
        ),
    }
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    all_metrics = {}
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None
    
    print("\n[5/5] Running 5-Fold StratifiedGroupKFold Cross-Validation & Holdout Evaluation...")
    
    for name, clf in model_candidates.items():
        print(f"\n---> Evaluating {name}...")
        pipeline = Pipeline([
            ("features", feature_union),
            ("classifier", clf)
        ])
        
        cv_scores = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": [], "pr_auc": []}
        
        for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_train_val, y_train_val, groups=groups_train_val)):
            X_tr, y_tr = X_train_val[tr_idx], y_train_val[tr_idx]
            X_va, y_va = X_train_val[val_idx], y_train_val[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_va)
            probs = pipeline.predict_proba(X_va)[:, 1]
            
            p, r, _ = precision_recall_curve(y_va, probs)
            cv_scores["accuracy"].append(accuracy_score(y_va, preds))
            cv_scores["precision"].append(precision_score(y_va, preds, zero_division=0))
            cv_scores["recall"].append(recall_score(y_va, preds, zero_division=0))
            cv_scores["f1"].append(f1_score(y_va, preds, zero_division=0))
            cv_scores["roc_auc"].append(roc_auc_score(y_va, probs))
            cv_scores["pr_auc"].append(auc(r, p))
            
        # Fit on full train_val and evaluate on holdout test
        t0 = time.time()
        pipeline.fit(X_train_val, y_train_val)
        fit_duration = time.time() - t0
        
        test_preds = pipeline.predict(X_test)
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        
        test_p, test_r, _ = precision_recall_curve(y_test, test_probs)
        cm = confusion_matrix(y_test, test_preds).tolist()
        
        holdout_metrics = {
            "accuracy": float(accuracy_score(y_test, test_preds)),
            "precision": float(precision_score(y_test, test_preds, zero_division=0)),
            "recall": float(recall_score(y_test, test_preds, zero_division=0)),
            "f1": float(f1_score(y_test, test_preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_probs)),
            "pr_auc": float(auc(test_r, test_p)),
            "confusion_matrix": cm,
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1]
        }
        
        cv_summary_mean = {k: float(np.mean(v)) for k, v in cv_scores.items()}
        cv_summary_std = {k: float(np.std(v)) for k, v in cv_scores.items()}
        
        print(f"      CV F1:      {cv_summary_mean['f1']*100:.2f}% ± {cv_summary_std['f1']*100:.2f}%")
        print(f"      Holdout Acc: {holdout_metrics['accuracy']*100:.2f}%")
        print(f"      Holdout Prec:{holdout_metrics['precision']*100:.2f}%")
        print(f"      Holdout Rec: {holdout_metrics['recall']*100:.2f}%")
        print(f"      Holdout F1:  {holdout_metrics['f1']*100:.2f}%")
        print(f"      Holdout AUC: {holdout_metrics['roc_auc']:.4f}")
        print(f"      Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
        
        all_metrics[name] = {
            "cv_mean": cv_summary_mean,
            "cv_std": cv_summary_std,
            "holdout": holdout_metrics,
            "fit_time_seconds": round(fit_duration, 2)
        }
        
        if holdout_metrics["f1"] > best_f1:
            best_f1 = holdout_metrics["f1"]
            best_model_name = name
            best_pipeline = pipeline

    # Save best model
    print(f"\n=======================================================")
    print(f"BEST PERFORMING MODEL ON HOLDOUT: {best_model_name} (F1: {best_f1*100:.2f}%)")
    print(f"=======================================================")
    
    prod_model_path = "models/live_model.joblib"
    joblib.dump(best_pipeline, prod_model_path)
    file_size_mb = os.path.getsize(prod_model_path) / (1024 * 1024)
    print(f"Saved production pipeline to '{prod_model_path}' ({file_size_mb:.2f} MB)")
    assert file_size_mb < 50.0, f"Error: Model file size exceeds 50 MB limit! ({file_size_mb:.2f} MB)"
    
    # Save reports/metrics.json
    metrics_record = {
        "best_model": best_model_name,
        "dataset_info": {
            "raw_samples": len(df),
            "deduplicated_samples": len(df_clean),
            "unique_registered_domains": num_domains,
            "train_val_samples": len(X_train_val),
            "holdout_test_samples": len(X_test),
        },
        "models": all_metrics
    }
    
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics_record, f, indent=2)
    print("Saved 'reports/metrics.json'")
    
    # Save reports/metrics.md
    with open("reports/metrics.md", "w") as f:
        f.write("# Model Evaluation & Benchmark Report\n\n")
        f.write("## Evaluation Methodology\n")
        f.write("- **Grouping:** Registered domain extracted using offline `tldextract`.\n")
        f.write("- **Holdout Split:** `GroupShuffleSplit` (80% train+val, 20% holdout test) with 0 domain overlap.\n")
        f.write("- **Cross-Validation:** 5-Fold `StratifiedGroupKFold` on the training set.\n")
        f.write("- **Features (Tier 1):** 40 lexical features (StandardScaled) + Character 3-5 gram TF-IDF (1000 features).\n\n")
        f.write("## Benchmark Comparison Table\n\n")
        f.write("| Model | 5-Fold CV F1 | Holdout Accuracy | Holdout Precision | Holdout Recall | Holdout F1 | Holdout ROC-AUC | Holdout PR-AUC | TN / FP / FN / TP |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for m_name, m_data in all_metrics.items():
            cv_f1 = f"{m_data['cv_mean']['f1']*100:.2f}% ± {m_data['cv_std']['f1']*100:.2f}%"
            h = m_data["holdout"]
            cm_str = f"{h['tn']} / {h['fp']} / {h['fn']} / {h['tp']}"
            f.write(f"| **{m_name}** | {cv_f1} | {h['accuracy']*100:.2f}% | {h['precision']*100:.2f}% | {h['recall']*100:.2f}% | **{h['f1']*100:.2f}%** | {h['roc_auc']:.4f} | {h['pr_auc']:.4f} | {cm_str} |\n")
        f.write(f"\n**Production Model Selected:** `{best_model_name}` (Size: {file_size_mb:.2f} MB, Threshold: 0.50 default)\n")
    print("Saved 'reports/metrics.md'")
    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
