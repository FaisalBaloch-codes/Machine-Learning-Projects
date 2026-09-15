# Dark Pool Signal — Final Report

> Trained and evaluated on real FINRA OTC Transparency (dark pool) data and real Yahoo Finance OHLCV price data. No synthetic data used.

## 1. Dataset Summary
- Test period rows: 1,564
- Abnormal-move base rate (test): 13.24%
- Label definition: |forward 1-week return| > 1.5x trailing 12-week realized volatility
- Naive baseline accuracy (always predicts 'normal'): 86.76%

## 2. Full Model x Threshold Comparison (Held-Out Chronological Test Set)
| model                 | threshold_type               |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:----------------------|:-----------------------------|-----------:|------------:|---------:|-------:|----------:|
| Naive Baseline        | n/a (always 0)               |     0.8676 |      0      |   0      | 0      |  nan      |
| Logistic Regression   | 0.5 (default)                |     0.8676 |      0      |   0      | 0      |    0.6065 |
| Logistic Regression   | F1-optimal (validation)      |     0.5697 |      0.1613 |   0.5362 | 0.248  |    0.6065 |
| Logistic Regression   | Accuracy-target (validation) |     0.8159 |      0.176  |   0.1063 | 0.1325 |    0.6065 |
| XGBoost               | 0.5 (default)                |     0.8676 |      0      |   0      | 0      |    0.6114 |
| XGBoost               | F1-optimal (validation)      |     0.626  |      0.1786 |   0.5072 | 0.2642 |    0.6114 |
| XGBoost               | Accuracy-target (validation) |     0.8504 |      0.2545 |   0.0676 | 0.1069 |    0.6114 |
| Ensemble (LR+XGB avg) | 0.5 (default)                |     0.8676 |      0      |   0      | 0      |    0.6127 |
| Ensemble (LR+XGB avg) | F1-optimal (validation)      |     0.6765 |      0.1785 |   0.401  | 0.247  |    0.6127 |
| Ensemble (LR+XGB avg) | Accuracy-target (validation) |     0.8408 |      0.2    |   0.0676 | 0.1011 |    0.6127 |

> **Why three thresholds per model?** With abnormal moves at only ~13.2% of weeks, a model can trivially exceed **86.8% accuracy** simply by predicting 'normal' almost always -- that's what the naive baseline does, with zero forecasting value. To show the real trade-off transparently, each model is reported at: **0.5** (arbitrary default), **F1-optimal** (best balance of precision/recall, tuned on the validation set), and **accuracy-target** (the highest accuracy achievable on the validation set while keeping recall at or above 10%, so it isn't just predicting nothing). See `eval_05_accuracy_recall_tradeoff.png` for the full curve -- accuracy and recall move in opposite directions as the threshold rises, and any single number quoted without its threshold and recall alongside it is not a complete picture of model quality.

## 3. Key Finding
**Ensemble (LR+XGB avg)** achieved the highest ROC-AUC (**0.613**, vs. 0.500 for random guessing), indicating the strongest ranking ability among the models tested. ROC-AUC is threshold-independent and is the primary metric for judging real predictive skill on this rare-event problem; the accuracy-target rows above show what happens to precision/recall if the decision threshold is pushed to chase a high accuracy number instead.

## 4. Files Produced
- `outputs/charts/eda_01..05_*.png` — exploratory analysis
- `outputs/charts/eval_01_roc_pr_curves.png` — ROC / Precision-Recall curves
- `outputs/charts/eval_02_metric_comparison.png` — all models x all thresholds
- `outputs/charts/eval_03_confusion_matrices.png` — confusion matrices (F1-optimal threshold)
- `outputs/charts/eval_04_feature_importance_both_models.png` — LR coefficients + XGBoost importances
- `outputs/charts/eval_05_accuracy_recall_tradeoff.png` — accuracy vs. recall by threshold
- `outputs/reports/evaluation_metrics.csv`, `chosen_thresholds.csv`, `threshold_scan_*.csv`
- `outputs/reports/model_predictions_test.csv` — row-level test predictions, all models/thresholds
- `outputs/models/logistic_regression.joblib`, `xgboost_model.joblib`