# 🛡️ CyberShield AI: Enterprise Phishing URL Detection System

An enterprise-grade, leak-free machine learning platform and interactive web application for detecting phishing websites and malicious URLs.

---

## 📌 Executive Summary

Many open-source phishing detection models suffer from severe data leakage:
1. **Random train/test splits** instead of domain-grouped splits (causing identical domains to appear in both sets).
2. **Duplicate raw URLs** that artificially inflate test accuracy to ~97–99% through rote memorization.
3. **Train-serve skew** where offline models rely on precomputed features (e.g. `domain_age`, requiring slow live WHOIS lookups at runtime) while the serving app computes different lexical metrics.

**CyberShield AI completely re-engineers this pipeline:**
- **Zero Leakage Evaluation:** Full URL normalization, deduplication, and strictly domain-grouped holdout partitioning (`GroupShuffleSplit` by registered domain using offline `tldextract`).
- **Single Source of Truth Feature Pipeline:** 40 Tier-1 lexical/statistical features combined with character 3–5 gram TF-IDF vectorization inside a single scikit-learn `Pipeline`. Zero external network calls at inference.
- **Robust Model Engineering:** LightGBM classifier with threshold optimization derived from out-of-fold validation, paired domain-clustered bootstrap confidence intervals, and local SHAP decision explanations.
- **Lightweight & Fast:** Production model serialized under **0.75 MB** (well below the 50 MB limit) with sub-10ms inference.

---

## 📊 Single Source of Truth Benchmark Results

All holdout metrics are computed from raw URL strings on the unseen 2,098-URL holdout test set (1,327 registered domains, **0% domain overlap**) and derived directly from [`reports/holdout_predictions.csv`](reports/holdout_predictions.csv).

### 1. Model Evolution & Honest Comparison

| Model Pipeline | Holdout Split Condition | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Rec @ 10% FPR | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Old Initial Model** *(Leaked random split)* | Same 2,098 URLs *(Overlaps training data)* | 97.71% | 97.04% | 97.68% | 97.36% | 0.9967 | 99.23% | 0.0239 |
| **Clean 18-Feature RF Baseline** | Domain-Grouped Holdout *(2,098 URLs)* | 86.75% | 85.44% | 83.55% | 84.49% | 0.9331 | 82.45% | 0.0994 |
| **New Final Pipeline** *(Default th=0.52)* | Domain-Grouped Holdout *(2,098 URLs)* | **88.37%** | **83.98%** | **90.29%** | **87.02%** | **0.9617** | **89.18%** | **0.0796** |
| **New Final Pipeline** *(High-Recall th=0.32)* | Domain-Grouped Holdout *(2,098 URLs)* | 86.56% | 78.73% | **94.37%** | 85.84% | **0.9617** | **89.18%** | **0.0796** |

> **Key Takeaway:** The old model's apparent "97.7%" accuracy was an illusion resulting from random train/test splits that leaked identical domains into test folds. On an honest domain-separated holdout, a clean 18-feature baseline achieves only 86.75% accuracy (0.9331 ROC-AUC). Our new pipeline delivers an honest, leak-free gain of **+2.86 AUC points** and **+6.73% Recall @ 10% FPR** with statistically significant bootstrap confidence bounds.

### 2. Paired Domain-Cluster Bootstrap (1,000 Resamples: New vs. 18-Feature Baseline)

- **$\Delta$ ROC-AUC:** `+0.0284` *(95% CI: [+0.0142, +0.0449])*
- **$\Delta$ PR-AUC:** `+0.0238` *(95% CI: [+0.0060, +0.0421])*
- **$\Delta$ Recall @ 10% FPR:** `+6.77%` *(95% CI: [+0.99%, +12.83%])*

*All confidence intervals strictly exclude zero, proving statistically significant superiority.*

### 3. Operating Point Comparison Across Decision Thresholds

| Threshold | Operating Target | ROC-AUC | PR-AUC | Brier Score | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TN, FP, FN, TP) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.32** | High-Recall *(OOF Recall 95.1%; Holdout 94.4%)* | 0.9617 | 0.9485 | 0.0796 | 86.56% | 78.73% | **94.37%** | 85.84% | TN=961, FP=231, FN=51, TP=855 |
| **0.36** | Historical Security Threshold | 0.9617 | 0.9485 | 0.0796 | 86.80% | 79.42% | 93.71% | 85.97% | TN=972, FP=220, FN=57, TP=849 |
| **0.49** | Historical Balanced Threshold | 0.9617 | 0.9485 | 0.0796 | 88.27% | 83.27% | 91.17% | 87.04% | TN=1026, FP=166, FN=80, TP=826 |
| **0.50** | Standard Default Threshold | 0.9617 | 0.9485 | 0.0796 | 88.23% | 83.45% | 90.73% | 86.94% | TN=1029, FP=163, FN=84, TP=822 |
| **0.52** | **Default (Balanced F1-Optimal)** | 0.9617 | 0.9485 | 0.0796 | **88.37%** | **83.98%** | 90.29% | **87.02%** | TN=1036, FP=156, FN=88, TP=818 |

*Note: ROC-AUC is mathematically identical (0.9617) across all thresholds because ROC-AUC measures ranking discrimination of the probability vector across all operating points.*

---

## 🔬 Source-Bias Experiment (Scheme & WWW Stripping)

To test for dataset collection artifacts (e.g., legitimate URLs scraped with `http://www.` and phishing URLs scraped as raw hostnames), we conducted an ablation stripping scheme (`https?://`) and leading `www.` before both lexical feature extraction and TF-IDF:

- **Standard Pipeline:** ROC-AUC: `0.9617`, PR-AUC: `0.9485`, F1: `0.8694`, Brier: `0.0796`
- **Stripped Pipeline:** ROC-AUC: `0.9567`, PR-AUC: `0.9390`, F1: `0.8679`, Brier: `0.0841`
- **Paired Bootstrap Difference:**
  - $\Delta$ ROC-AUC: `-0.0050` *(95% CI: [-0.0126, +0.0016], not statistically significant)*
- **SHAP N-gram Analysis:**
  - In the unstripped pipeline, n-grams like `'//www'`, `'//ww'`, and `'/www.'` ranked among the top 10 SHAP features (revealing dataset collection artifacts).
  - In the stripped pipeline, those artifacts completely disappeared, and the top features shifted to true semantic indicators (`'mail'`, `'/wp'`, `'wiki'`, `'wp-'`, `'app'`, `'cure'`, `'.pdf'`, `'.php'`).

---

## 🛡️ Security & Anti-SSRF Protections

The platform features optional live network checks (DNS resolution, SSL certificate inspection, WHOIS age calculation). To prevent Server-Side Request Forgery (SSRF) and hanging processes:
- **Private IP Blocking:** Resolves hostnames and rejects loopback (`127.0.0.0/8`, `::1`), private ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), and link-local addresses.
- **Strict Timeouts:** All socket connections, SSL handshakes, and WHOIS queries enforce strict 3.0-second timeouts.
- **Zero-Network Default:** The core ML model never makes network calls during feature extraction or prediction.

---

## 🚀 Quickstart & Usage

### 1. Installation

```bash
git clone https://github.com/your-username/phishing-url-detection.git
cd phishing-url-detection

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/macOS

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Run Test Suite

```bash
pytest tests/
```

### 3. Launch Streamlit Web Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 4. Reproduce Model Training & Holdout Evaluation

```bash
# Train final production pipeline on 100% Train+Val
python scripts/train_final.py

# Run Single Source of Truth holdout evaluation
python scripts/evaluate_final.py
```

---

## 📁 Repository Structure

```
├── app.py                       # Main Streamlit application entrypoint
├── conftest.py                  # Pytest configuration and path resolution
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development & testing dependencies
├── data/
│   ├── phishingURL.csv          # Base dataset (11,430 rows, 6,634 domains)
│   └── external/                # Directory for external validation feeds
├── models/
│   ├── live_model.joblib        # Final production LightGBM pipeline (0.75 MB)
│   └── threshold_config.json    # Operating threshold configuration
├── pages/
│   ├── home.py                  # Hero page with dynamic leak-free metrics
│   ├── single_scan.py           # Single URL scanner with SHAP & threat gauge
│   ├── bulk_scan.py             # CSV batch upload & download
│   └── analytics.py             # Model benchmarking & CV vs holdout charts
├── reports/
│   ├── holdout_predictions.csv  # Single Source of Truth holdout predictions
│   ├── metrics.json             # Dynamic metric storage
│   ├── metrics.md               # Markdown benchmark report
│   ├── ablation_study.md        # Feature & allowlist ablation study
│   ├── calibration_curve.png    # Probability reliability curves
│   ├── pr_roc_curve.png         # Precision-Recall & ROC curves
│   ├── shap_summary.png         # Global SHAP feature importance plot
│   └── error_analysis.md        # Top 25 FP/FN error analysis
├── scripts/
│   ├── train_final.py           # Production training script
│   ├── evaluate_final.py        # Single source of truth evaluation script
│   └── build_external_set.py    # External test set builder (anti-leakage)
├── tests/
│   └── test_feature_extraction.py # 14 pytest unit tests
└── utils/
    ├── feature_extraction.py    # 40 Tier-1 lexical features (Single Source of Truth)
    ├── predictor.py             # Unified inference engine
    ├── explainability.py        # Local & global SHAP explainers
    ├── dns_check.py             # SSRF-protected DNS checker
    ├── ssl_check.py             # SSRF-protected SSL validator
    ├── whois_check.py           # SSRF-protected WHOIS checker
    └── database.py              # SQLite history logging
```
