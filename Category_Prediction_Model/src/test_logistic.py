"""
Final Test - Logistic Regression Evaluation & Prediction Generation
=====================================================================

This script:
1. Loads the selected L1 Logistic Regression model and evaluates it on
   the untouched 1,000 test articles (validation_test_tfidf.csv).
2. Prints performance metrics (Accuracy, Macro F1, Weighted F1, Report, Confusion Matrix).
3. Generates results/L1_model_predictions.csv (L1 model + validation_test_data.csv/validation_test_tfidf.csv).
4. Generates results/L2_model_predictions.csv (L2 model + validation_test_data.csv/validation_test_tfidf.csv).
5. Generates results/log_regmodel_predictions.csv (Baseline model + test_data.csv/test_tfidf.csv).
"""

import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ---------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------

# Validation Test Data & Models (L1 and L2)
VAL_TEST_DATA_FILE = "data/validation_test_data.csv"
VAL_TEST_TFIDF_FILE = "data/validation_test_tfidf.csv"
L1_MODEL_FILE = "models/logistic_regression_validation_l1.pkl"
L2_MODEL_FILE = "models/logistic_regression_validation_l2.pkl"

# Baseline Test Data & Model
BASELINE_DATA_FILE = "data/test_data.csv"
BASELINE_TFIDF_FILE = "data/test_tfidf.csv"
BASELINE_MODEL_FILE = "models/logistic_regression_model.pkl"

# Output Prediction CSV Files
RESULTS_DIR = "results"
L1_PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "L1_model_predictions.csv")
L2_PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "L2_model_predictions.csv")
BASELINE_PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "log_regmodel_predictions.csv")


def main():
    print("=" * 65)
    print("FINAL TEST & PREDICTION GENERATION - LOGISTIC REGRESSION")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Evaluate L1 Model on Untouched Test Data
    # ---------------------------------------------------------------
    print("\nLoading L1 validation model and untouched test data...")
    val_test_df = pd.read_csv(VAL_TEST_DATA_FILE, low_memory=True)
    val_test_tfidf = pd.read_csv(VAL_TEST_TFIDF_FILE, low_memory=True)
    l1_model = joblib.load(L1_MODEL_FILE)

    val_tfidf_cols = [col for col in val_test_tfidf.columns if col.startswith("tfidf_")]
    X_val_test = val_test_tfidf[val_tfidf_cols]
    y_val_test = val_test_df["Category"]

    print(f"Test rows: {len(val_test_df)}")
    print("\nMaking predictions with L1 model...")
    l1_y_pred = l1_model.predict(X_val_test)

    accuracy = accuracy_score(y_val_test, l1_y_pred)
    macro_f1 = f1_score(y_val_test, l1_y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val_test, l1_y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 65)
    print("FINAL TEST PERFORMANCE (L1 MODEL)")
    print("=" * 65)

    print(f"\nAccuracy   : {accuracy:.4f}")
    print(f"Accuracy   : {accuracy * 100:.2f}%")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:")
    print("-" * 65)
    print(classification_report(y_val_test, l1_y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print("-" * 65)
    labels = l1_model.classes_
    cm = confusion_matrix(y_val_test, l1_y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # ---------------------------------------------------------------
    # 2. Generate L1_model_predictions.csv
    # ---------------------------------------------------------------
    l1_predictions_df = pd.DataFrame({
        "Title": val_test_df["Title"],
        "Actual_Category": val_test_df["Category"],
        "Predicted_Category": l1_y_pred
    })
    l1_predictions_df.to_csv(L1_PREDICTIONS_FILE, index=False)
    print(f"\nSaved L1 predictions to: {L1_PREDICTIONS_FILE} ({len(l1_predictions_df)} rows)")

    # ---------------------------------------------------------------
    # 3. Generate L2_model_predictions.csv
    # ---------------------------------------------------------------
    print("\nGenerating L2 model predictions...")
    l2_model = joblib.load(L2_MODEL_FILE)
    l2_y_pred = l2_model.predict(X_val_test)

    l2_predictions_df = pd.DataFrame({
        "Title": val_test_df["Title"],
        "Actual_Category": val_test_df["Category"],
        "Predicted_Category": l2_y_pred
    })
    l2_predictions_df.to_csv(L2_PREDICTIONS_FILE, index=False)
    print(f"Saved L2 predictions to: {L2_PREDICTIONS_FILE} ({len(l2_predictions_df)} rows)")

    # ---------------------------------------------------------------
    # 4. Generate log_regmodel_predictions.csv (Original Baseline)
    # ---------------------------------------------------------------
    print("\nGenerating Original Baseline Logistic Regression predictions...")
    base_test_df = pd.read_csv(BASELINE_DATA_FILE, low_memory=True)
    base_test_tfidf = pd.read_csv(BASELINE_TFIDF_FILE, low_memory=True)
    base_model = joblib.load(BASELINE_MODEL_FILE)

    base_tfidf_cols = [col for col in base_test_tfidf.columns if col.startswith("tfidf_")]
    X_base_test = base_test_tfidf[base_tfidf_cols]

    base_y_pred = base_model.predict(X_base_test)

    base_predictions_df = pd.DataFrame({
        "Title": base_test_df["Title"],
        "Actual_Category": base_test_df["Category"],
        "Predicted_Category": base_y_pred
    })
    base_predictions_df.to_csv(BASELINE_PREDICTIONS_FILE, index=False)
    print(f"Saved Baseline predictions to: {BASELINE_PREDICTIONS_FILE} ({len(base_predictions_df)} rows)")

    print("\n" + "=" * 65)
    print("FINAL TEST & PREDICTION GENERATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()