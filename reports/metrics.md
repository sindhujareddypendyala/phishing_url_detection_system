# Production Model Holdout Evaluation (Single Source of Truth)

All metrics are derived directly from `reports/holdout_predictions.csv` on the unseen holdout set (2,098 URLs across 1,327 registered domains).

## 1. Operating Point Comparison Across Decision Thresholds

|   threshold |   roc_auc |   pr_auc |   brier |   accuracy |   precision |   recall |     f1 |   tn |   fp |   fn |   tp |
|------------:|----------:|---------:|--------:|-----------:|------------:|---------:|-------:|-----:|-----:|-----:|-----:|
|        0.32 |    0.9617 |   0.9485 |  0.0796 |     0.8656 |      0.7873 |   0.9437 | 0.8584 |  961 |  231 |   51 |  855 |
|        0.36 |    0.9617 |   0.9485 |  0.0796 |     0.868  |      0.7942 |   0.9371 | 0.8597 |  972 |  220 |   57 |  849 |
|        0.49 |    0.9617 |   0.9485 |  0.0796 |     0.8827 |      0.8327 |   0.9117 | 0.8704 | 1026 |  166 |   80 |  826 |
|        0.5  |    0.9617 |   0.9485 |  0.0796 |     0.8823 |      0.8345 |   0.9073 | 0.8694 | 1029 |  163 |   84 |  822 |
|        0.52 |    0.9617 |   0.9485 |  0.0796 |     0.8837 |      0.8398 |   0.9029 | 0.8702 | 1036 |  156 |   88 |  818 |

*Note: ROC-AUC is mathematically identical (0.9617) across all thresholds because it evaluates overall ranking quality.*

## 2. Model Evolution & Baseline Comparison

| Model Pipeline                           | Holdout Split                            | Accuracy   | Precision   | Recall   | F1-Score   |   ROC-AUC | Rec@10%FPR   |   Brier |
|:-----------------------------------------|:-----------------------------------------|:-----------|:------------|:---------|:-----------|----------:|:-------------|--------:|
| Old Initial Model (Leaked random split)  | Same 2,098 URLs (Overlaps training data) | 97.71%     | 97.04%      | 97.68%   | 97.36%     |    0.9967 | 99.23%       |  0.0239 |
| Clean 18-Feature RF Baseline             | Domain-Grouped Holdout (2,098 URLs)      | 86.75%     | 85.44%      | 83.55%   | 84.49%     |    0.9331 | 82.45%       |  0.0994 |
| New Final Pipeline (Default th=0.52)     | Domain-Grouped Holdout (2,098 URLs)      | 88.37%     | 83.98%      | 90.29%   | 87.02%     |    0.9617 | 89.18%       |  0.0796 |
| New Final Pipeline (High-Recall th=0.32) | Domain-Grouped Holdout (2,098 URLs)      | 86.56%     | 78.73%      | 94.37%   | 85.84%     |    0.9617 | 89.18%       |  0.0796 |

## 3. Paired Bootstrap Comparison (New Pipeline vs 18-Feature Baseline)

- **Delta ROC-AUC:** +0.0284 (95% CI: [+0.0142, +0.0449])
- **Delta PR-AUC:** +0.0238 (95% CI: [+0.0060, +0.0421])
- **Delta Recall @ 10% FPR:** +6.77% (95% CI: [+0.99%, +12.83%])
