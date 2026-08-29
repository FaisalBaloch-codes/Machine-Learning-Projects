"""
SVM Classification Model (RBF Kernel) for News Category Classification
========================================================================

This script trains a Support Vector Machine with an RBF (Radial Basis
Function) kernel on already-computed TF-IDF features and evaluates it
category by category.

IMPORTANT:
- TF-IDF has already been calculated in tfidf_calculation.py.
- This script does NOT recompute TF-IDF and does NOT do feature engineering.
- It only loads train_tfidf.csv / test_tfidf.csv, scales the features,
  trains an SVC(kernel="rbf"), and evaluates the results.
"""

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
DATA_DIR = "data"

TRAIN_TFIDF_PATH = f"{DATA_DIR}/train_tfidf.csv"
TEST_TFIDF_PATH = f"{DATA_DIR}/test_tfidf.csv"
TEST_DATA_PATH = f"{DATA_DIR}/test_data.csv"

PREDICTIONS_OUTPUT_PATH = f"{DATA_DIR}/SVM_RBF_predictions.csv"
CATEGORY_METRICS_OUTPUT_PATH = f"{DATA_DIR}/SVM_RBF_category_metrics.csv"
OVERALL_SUMMARY_OUTPUT_PATH = f"{DATA_DIR}/SVM_RBF_overall_summary.csv"

TARGET_COLUMN = "Category"


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
print_header("SVM CLASSIFICATION - RBF KERNEL")

train_df = pd.read_csv(TRAIN_TFIDF_PATH)
test_df = pd.read_csv(TEST_TFIDF_PATH)

if TARGET_COLUMN not in train_df.columns:
    raise ValueError(f"'{TARGET_COLUMN}' column not found in {TRAIN_TFIDF_PATH}")
if TARGET_COLUMN not in test_df.columns:
    raise ValueError(f"'{TARGET_COLUMN}' column not found in {TEST_TFIDF_PATH}")

# Feature columns = every column except the target (Category)
feature_columns = [c for c in train_df.columns if c != TARGET_COLUMN]

# Make sure train and test have identical feature columns, in the same order
test_feature_columns = [c for c in test_df.columns if c != TARGET_COLUMN]
if feature_columns != test_feature_columns:
    raise ValueError(
        "Train and test feature columns do not match. "
        "Check that train_tfidf.csv and test_tfidf.csv were generated "
        "with the exact same vocabulary/column order."
    )

X_train = train_df[feature_columns]
y_train = train_df[TARGET_COLUMN]

X_test = test_df[feature_columns]
y_test = test_df[TARGET_COLUMN]

categories = sorted(y_train.unique())

print("\nDataset information:")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Number of features: {len(feature_columns)}")
print(f"Number of categories: {len(categories)}")

print("\nKernel: RBF")
print('SVM model: SVC(kernel="rbf")')

# ------------------------------------------------------------------
# 2. DATA QUALITY CHECKS
# ------------------------------------------------------------------
# Missing values
train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()
if train_missing > 0 or test_missing > 0:
    print(f"\nWARNING: Missing values found -> train: {train_missing}, test: {test_missing}")
else:
    print("\nMissing values check: OK (no missing values found)")

# Infinite values
train_inf = np.isinf(X_train.values).sum()
test_inf = np.isinf(X_test.values).sum()
if train_inf > 0 or test_inf > 0:
    print(f"WARNING: Infinite values found -> train: {train_inf}, test: {test_inf}")
else:
    print("Infinite values check: OK (no infinite values found)")

# Non-numeric feature columns
non_numeric_train = [c for c in feature_columns if not np.issubdtype(X_train[c].dtype, np.number)]
non_numeric_test = [c for c in feature_columns if not np.issubdtype(X_test[c].dtype, np.number)]
if non_numeric_train or non_numeric_test:
    raise ValueError(
        f"Non-numeric feature columns detected. "
        f"Train: {non_numeric_train}, Test: {non_numeric_test}"
    )
else:
    print("Non-numeric column check: OK (all feature columns are numeric)")

# ------------------------------------------------------------------
# 3. CATEGORY DISTRIBUTION
# ------------------------------------------------------------------
print_header("CATEGORY DISTRIBUTION")

print("\nTraining:")
print(y_train.value_counts())

print("\nTesting:")
print(y_test.value_counts())

# ------------------------------------------------------------------
# 4. FEATURE SCALING (fit on train only, to avoid data leakage)
# ------------------------------------------------------------------
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 5. TRAIN THE SVM MODEL (RBF KERNEL)
# ------------------------------------------------------------------
print_header("MODEL TRAINING")

print("\nTraining SVM...")

model = SVC(kernel="rbf", random_state=42)
model.fit(X_train_scaled, y_train)

print("Model training completed.")

# ------------------------------------------------------------------
# 6. PREDICT ON TEST SET
# ------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

# ------------------------------------------------------------------
# 7. PREDICTIONS CSV (Title + Actual + Predicted, aligned)
# ------------------------------------------------------------------
test_data_df = pd.read_csv(TEST_DATA_PATH)

if len(test_data_df) != len(y_test):
    raise ValueError(
        f"Row count mismatch between {TEST_DATA_PATH} ({len(test_data_df)} rows) "
        f"and test_tfidf.csv ({len(y_test)} rows). Cannot safely align Title column."
    )

predictions_df = pd.DataFrame({
    "Title": test_data_df["Title"].values,
    "Actual_Category": y_test.values,
    "Predicted_Category": y_pred,
})

predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

# ------------------------------------------------------------------
# 8. OVERALL METRICS
# ------------------------------------------------------------------
overall_accuracy = accuracy_score(y_test, y_pred)

macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0
)

print_header("OVERALL RESULTS")

print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")

print(f"\nMacro Precision: {macro_precision * 100:.2f}%")
print(f"Macro Recall: {macro_recall * 100:.2f}%")
print(f"Macro F1-Score: {macro_f1 * 100:.2f}%")

print(f"\nWeighted Precision: {weighted_precision * 100:.2f}%")
print(f"Weighted Recall: {weighted_recall * 100:.2f}%")
print(f"Weighted F1-Score: {weighted_f1 * 100:.2f}%")

# ------------------------------------------------------------------
# 9. CATEGORY-WISE (ONE-VS-REST) METRICS
# ------------------------------------------------------------------
y_test_arr = y_test.values
y_pred_arr = np.array(y_pred)

category_rows = []

for category in categories:
    true_binary = (y_test_arr == category)
    pred_binary = (y_pred_arr == category)

    cat_accuracy = accuracy_score(true_binary, pred_binary)
    cat_precision = precision_score(true_binary, pred_binary, zero_division=0)
    cat_recall = recall_score(true_binary, pred_binary, zero_division=0)
    cat_f1 = f1_score(true_binary, pred_binary, zero_division=0)
    cat_support = int(true_binary.sum())

    category_rows.append({
        "Category": category,
        "Accuracy": cat_accuracy,
        "Precision": cat_precision,
        "Recall": cat_recall,
        "F1-Score": cat_f1,
        "Support": cat_support,
    })

category_metrics_df = pd.DataFrame(category_rows)

# Sort by F1-Score, highest first
category_metrics_df = category_metrics_df.sort_values(
    by="F1-Score", ascending=False
).reset_index(drop=True)

category_metrics_df.to_csv(CATEGORY_METRICS_OUTPUT_PATH, index=False)

print_header("CATEGORY-WISE RESULTS")
print()
display_df = category_metrics_df.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    display_df[col] = display_df[col].map(lambda x: f"{x * 100:.2f}%")
print(display_df.to_string(index=False))

# ------------------------------------------------------------------
# 10. BEST / WORST CATEGORY PER METRIC
# ------------------------------------------------------------------
def best_worst(metric_column):
    best_row = category_metrics_df.loc[category_metrics_df[metric_column].idxmax()]
    worst_row = category_metrics_df.loc[category_metrics_df[metric_column].idxmin()]
    return best_row["Category"], best_row[metric_column], worst_row["Category"], worst_row[metric_column]

best_acc_cat, best_acc_val, worst_acc_cat, worst_acc_val = best_worst("Accuracy")
best_prec_cat, best_prec_val, worst_prec_cat, worst_prec_val = best_worst("Precision")
best_rec_cat, best_rec_val, worst_rec_cat, worst_rec_val = best_worst("Recall")
best_f1_cat, best_f1_val, worst_f1_cat, worst_f1_val = best_worst("F1-Score")

print_header("BEST / WORST CATEGORY")

print(f"\nHighest Accuracy: {best_acc_cat} ({best_acc_val * 100:.2f}%)")
print(f"Lowest Accuracy: {worst_acc_cat} ({worst_acc_val * 100:.2f}%)")

print(f"\nHighest Precision: {best_prec_cat} ({best_prec_val * 100:.2f}%)")
print(f"Lowest Precision: {worst_prec_cat} ({worst_prec_val * 100:.2f}%)")

print(f"\nHighest Recall: {best_rec_cat} ({best_rec_val * 100:.2f}%)")
print(f"Lowest Recall: {worst_rec_cat} ({worst_rec_val * 100:.2f}%)")

print(f"\nHighest F1-Score: {best_f1_cat} ({best_f1_val * 100:.2f}%)")
print(f"Lowest F1-Score: {worst_f1_cat} ({worst_f1_val * 100:.2f}%)")

# ------------------------------------------------------------------
# 11. CLASSIFICATION REPORT
# ------------------------------------------------------------------
print_header("CLASSIFICATION REPORT")
print()
print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------------------------------------------
# 12. OVERALL SUMMARY CSV
# ------------------------------------------------------------------
summary_rows = [
    ("Overall Accuracy", overall_accuracy),
    ("Macro Precision", macro_precision),
    ("Macro Recall", macro_recall),
    ("Macro F1-Score", macro_f1),
    ("Weighted Precision", weighted_precision),
    ("Weighted Recall", weighted_recall),
    ("Weighted F1-Score", weighted_f1),
    ("Highest Accuracy Category", f"{best_acc_cat} ({best_acc_val * 100:.2f}%)"),
    ("Lowest Accuracy Category", f"{worst_acc_cat} ({worst_acc_val * 100:.2f}%)"),
    ("Highest Precision Category", f"{best_prec_cat} ({best_prec_val * 100:.2f}%)"),
    ("Lowest Precision Category", f"{worst_prec_cat} ({worst_prec_val * 100:.2f}%)"),
    ("Highest Recall Category", f"{best_rec_cat} ({best_rec_val * 100:.2f}%)"),
    ("Lowest Recall Category", f"{worst_rec_cat} ({worst_rec_val * 100:.2f}%)"),
    ("Highest F1-Score Category", f"{best_f1_cat} ({best_f1_val * 100:.2f}%)"),
    ("Lowest F1-Score Category", f"{worst_f1_cat} ({worst_f1_val * 100:.2f}%)"),
]

overall_summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
overall_summary_df.to_csv(OVERALL_SUMMARY_OUTPUT_PATH, index=False)

# ------------------------------------------------------------------
# 13. OUTPUT FILES
# ------------------------------------------------------------------
print_header("OUTPUT FILES")

print(f"\n{PREDICTIONS_OUTPUT_PATH}")
print(f"{CATEGORY_METRICS_OUTPUT_PATH}")
print(f"{OVERALL_SUMMARY_OUTPUT_PATH}")
print()