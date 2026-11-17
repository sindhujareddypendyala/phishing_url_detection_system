# Model Evaluation & Benchmark Report

## Evaluation Methodology
- **Grouping:** Registered domain extracted using offline `tldextract`.
- **Holdout Split:** `GroupShuffleSplit` (80% train+val, 20% holdout test) with 0 domain overlap.
- **Cross-Validation:** 5-Fold `StratifiedGroupKFold` on the training set.
- **Features (Tier 1):** 40 lexical features (StandardScaled) + Character 3-5 gram TF-IDF (1000 features).

## Benchmark Comparison Table

| Model | 5-Fold CV F1 | Holdout Accuracy | Holdout Precision | Holdout Recall | Holdout F1 | Holdout ROC-AUC | Holdout PR-AUC | TN / FP / FN / TP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Baseline)** | 87.93% ± 2.33% | 83.98% | 77.62% | 88.41% | **82.66%** | 0.9378 | 0.9262 | 961 / 231 / 105 / 801 |
| **Logistic Regression** | 88.36% ± 3.29% | 88.13% | 84.76% | 88.41% | **86.55%** | 0.9526 | 0.9300 | 1048 / 144 / 105 / 801 |
| **LightGBM** | 91.22% ± 2.03% | 88.23% | 83.45% | 90.73% | **86.94%** | 0.9617 | 0.9485 | 1029 / 163 / 84 / 822 |
| **XGBoost** | 90.13% ± 1.75% | 86.65% | 82.54% | 87.64% | **85.01%** | 0.9557 | 0.9406 | 1024 / 168 / 112 / 794 |

**Production Model Selected:** `LightGBM` (Size: 0.75 MB, Threshold: 0.50 default)
