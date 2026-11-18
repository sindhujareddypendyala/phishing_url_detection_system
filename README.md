# 🛡️ CyberShield AI – Phishing URL Detection System

An intelligent machine learning-based web application that detects phishing and malicious URLs using advanced feature extraction and classification techniques.

## 📌 Overview

CyberShield AI helps users identify potentially dangerous URLs and protect themselves from phishing attacks.

Unlike traditional phishing detection systems, this project focuses on preventing data leakage during model training and ensuring reliable performance on previously unseen domains.

### Key Features

- **Accurate URL Detection:** Identifies legitimate and phishing URLs using machine learning.
- **Leak-Free Training:** Uses domain-based data splitting to prevent data leakage.
- **Advanced Feature Extraction:** Extracts 40 lexical and statistical URL features.
- **TF-IDF Analysis:** Uses character-level 3–5 gram TF-IDF features.
- **SHAP Explainability:** Explains why a URL is classified as suspicious.
- **Single URL Scanning:** Analyzes individual URLs.
- **Bulk URL Scanning:** Upload CSV files to analyze multiple URLs.
- **Security Protection:** Includes DNS, SSL, and WHOIS checks with SSRF protection.
- **Fast Predictions:** Performs predictions without external network requests.

## 📊 Model Performance

The final model uses a LightGBM classifier trained with a combined feature extraction and TF-IDF pipeline.

Evaluation was performed on 2,098 previously unseen URLs across 1,327 registered domains, with zero domain overlap between training and testing data.

| Metric | Performance |
|---|---:|
| Accuracy | **88.37%** |
| Precision | **83.98%** |
| Recall | **90.29%** |
| F1-Score | **87.02%** |
| ROC-AUC | **0.9617** |

The system also provides a high-recall detection option with **94.37% recall** to identify more potential phishing URLs.

## 🧠 Machine Learning Approach

1. URL preprocessing and normalization.
2. Duplicate URL removal.
3. Domain-based training and testing separation.
4. Extraction of 40 lexical and statistical features.
5. Character-level TF-IDF vectorization.
6. LightGBM model training and evaluation.
7. Decision threshold optimization.
8. SHAP-based model explanations.

## 🔐 Security Features

- Blocks private and loopback IP addresses during live network checks.
- Uses strict 3-second connection timeouts.
- Includes DNS and SSL certificate validation.
- Supports WHOIS-based domain age checks.
- Keeps the core prediction pipeline independent of external network calls.

## 🚀 Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | LightGBM, Scikit-learn |
| Feature Extraction | TF-IDF, URL Lexical Features |
| Explainable AI | SHAP |
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Testing | Pytest |
| Database | SQLite |

## 📂 Project Structure

```text
phishing-url-detection/
│
├── app.py
├── requirements.txt
├── requirements-dev.txt
│
├── data/
│   └── phishingURL.csv
│
├── models/
│   ├── live_model.joblib
│   └── threshold_config.json
│
├── pages/
│   ├── home.py
│   ├── single_scan.py
│   ├── bulk_scan.py
│   └── analytics.py
│
├── reports/
│   ├── holdout_predictions.csv
│   ├── metrics.json
│   ├── metrics.md
│   └── error_analysis.md
│
├── scripts/
│   ├── train_final.py
│   ├── evaluate_final.py
│   └── build_external_set.py
│
├── tests/
│   └── test_feature_extraction.py
│
└── utils/
    ├── feature_extraction.py
    ├── predictor.py
    ├── explainability.py
    ├── dns_check.py
    ├── ssl_check.py
    ├── whois_check.py
    └── database.py
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/phishing-url-detection.git

cd phishing-url-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Run Tests

```bash
pytest tests/
```

### 5. Launch the Application

```bash
streamlit run app.py
```

Open your browser and visit:

```text
http://localhost:8501
```

## 📈 Model Training and Evaluation

To train the production model:

```bash
python scripts/train_final.py
```

To evaluate model performance:

```bash
python scripts/evaluate_final.py
```

## 🎯 Project Objective

The main objective of CyberShield AI is to develop a reliable, explainable, and efficient phishing URL detection system that can identify malicious websites while minimizing data leakage and improving real-world detection performance.

## 👩‍💻 Author

**Sindhuja Pendyala**

B.Tech – Data Science


---

⭐ If you find this project useful, consider giving it a star!
