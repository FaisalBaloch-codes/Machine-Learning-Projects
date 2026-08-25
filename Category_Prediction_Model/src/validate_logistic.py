"""
News Classification - L1 vs L2 Logistic Regression
===================================================

This script:
1. Loads the NEW validation TF-IDF datasets.
2. Trains Logistic Regression with L1 regularization.
3. Trains Logistic Regression with L2 regularization.
4. Evaluates both models on the SAME validation data.
5. Compares Accuracy, Precision, Recall and F1.
6. Saves both trained models.
7. Saves a comparison CSV.

IMPORTANT:
- Original baseline files are NOT modified.
- Final test data is NOT used here.
- Validation data is used only for model comparison.
- Both models use exactly the same training/validation data.

Data:
    Training    = 3,599
    Validation  = 400
    Final Test  = 1,000 (untouched)

TF-IDF:
    min_df = 2
    max_df = 0.95

Models:
    Logistic Regression + L1
    Logistic Regression + L2
"""

# ---------------------------------------------------------------
# Imports
# ---------------------------------------------------------------

import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------------
# File paths
# ---------------------------------------------------------------

TRAIN_FILE = "data/train_validation_tfidf.csv"
VALIDATION_FILE = "data/validation_tfidf.csv"

L1_MODEL_FILE = "models/logistic_regression_validation_l1.pkl"
L2_MODEL_FILE = "models/logistic_regression_validation_l2.pkl"

COMPARISON_FILE = "results/logistic_l1_l2_comparison.csv"


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

RANDOM_STATE = 42
MAX_ITER = 1000


# ---------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------

print("=" * 70)
print("LOGISTIC REGRESSION - L1 vs L2 VALIDATION")
print("=" * 70)

print("\nLoading validation experiment data...")

train_df = pd.read_csv(TRAIN_FILE, low_memory=True)
validation_df = pd.read_csv(VALIDATION_FILE, low_memory=True)

print(f"Training rows   : {len(train_df)}")
print(f"Validation rows : {len(validation_df)}")


# ---------------------------------------------------------------
# Step 2: Separate features and target
# ---------------------------------------------------------------

feature_columns = [
    column
    for column in train_df.columns
    if column.startswith("tfidf_")
]

X_train = train_df[feature_columns]
y_train = train_df["Category"]

X_validation = validation_df[feature_columns]
y_validation = validation_df["Category"]

print(f"TF-IDF features : {len(feature_columns)}")


# ---------------------------------------------------------------
# Step 3: Create L1 Logistic Regression
# ---------------------------------------------------------------

print("\n" + "=" * 70)
print("L1 LOGISTIC REGRESSION")
print("=" * 70)

print("\nCreating L1 model...")

l1_model = LogisticRegression(
    l1_ratio=1,
    solver="saga",
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE
)


# ---------------------------------------------------------------
# Step 4: Train L1
# ---------------------------------------------------------------

print("Training L1 model...")

l1_model.fit(X_train, y_train)

print("L1 training completed.")


# ---------------------------------------------------------------
# Step 5: L1 predictions
# ---------------------------------------------------------------

print("Making L1 predictions...")

l1_predictions = l1_model.predict(X_validation)


# ---------------------------------------------------------------
# Step 6: L1 metrics
# ---------------------------------------------------------------

l1_accuracy = accuracy_score(
    y_validation,
    l1_predictions
)

l1_precision = precision_score(
    y_validation,
    l1_predictions,
    average="weighted",
    zero_division=0
)

l1_recall = recall_score(
    y_validation,
    l1_predictions,
    average="weighted",
    zero_division=0
)

l1_f1 = f1_score(
    y_validation,
    l1_predictions,
    average="weighted",
    zero_division=0
)

l1_macro_f1 = f1_score(
    y_validation,
    l1_predictions,
    average="macro",
    zero_division=0
)


# ---------------------------------------------------------------
# Step 7: L1 detailed report
# ---------------------------------------------------------------

print("\nL1 RESULTS")
print("-" * 70)

print(f"Accuracy       : {l1_accuracy:.4f} ({l1_accuracy * 100:.2f}%)")
print(f"Weighted F1    : {l1_f1:.4f}")
print(f"Macro F1       : {l1_macro_f1:.4f}")

print("\nL1 Classification Report:")
print(
    classification_report(
        y_validation,
        l1_predictions,
        zero_division=0
    )
)


# ---------------------------------------------------------------
# Step 8: Create L2 Logistic Regression
# ---------------------------------------------------------------

print("\n" + "=" * 70)
print("L2 LOGISTIC REGRESSION")
print("=" * 70)

print("\nCreating L2 model...")

l2_model = LogisticRegression(
    l1_ratio=0,
    solver="lbfgs",
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE
)


# ---------------------------------------------------------------
# Step 9: Train L2
# ---------------------------------------------------------------

print("Training L2 model...")

l2_model.fit(X_train, y_train)

print("L2 training completed.")


# ---------------------------------------------------------------
# Step 10: L2 predictions
# ---------------------------------------------------------------

print("Making L2 predictions...")

l2_predictions = l2_model.predict(X_validation)


# ---------------------------------------------------------------
# Step 11: L2 metrics
# ---------------------------------------------------------------

l2_accuracy = accuracy_score(
    y_validation,
    l2_predictions
)

l2_precision = precision_score(
    y_validation,
    l2_predictions,
    average="weighted",
    zero_division=0
)

l2_recall = recall_score(
    y_validation,
    l2_predictions,
    average="weighted",
    zero_division=0
)

l2_f1 = f1_score(
    y_validation,
    l2_predictions,
    average="weighted",
    zero_division=0
)

l2_macro_f1 = f1_score(
    y_validation,
    l2_predictions,
    average="macro",
    zero_division=0
)


# ---------------------------------------------------------------
# Step 12: L2 detailed report
# ---------------------------------------------------------------

print("\nL2 RESULTS")
print("-" * 70)

print(f"Accuracy       : {l2_accuracy:.4f} ({l2_accuracy * 100:.2f}%)")
print(f"Weighted F1    : {l2_f1:.4f}")
print(f"Macro F1       : {l2_macro_f1:.4f}")

print("\nL2 Classification Report:")
print(
    classification_report(
        y_validation,
        l2_predictions,
        zero_division=0
    )
)


# ---------------------------------------------------------------
# Step 13: Confusion matrices
# ---------------------------------------------------------------

labels = l2_model.classes_

l1_cm = confusion_matrix(
    y_validation,
    l1_predictions,
    labels=labels
)

l2_cm = confusion_matrix(
    y_validation,
    l2_predictions,
    labels=labels
)


print("\n" + "=" * 70)
print("L1 CONFUSION MATRIX")
print("=" * 70)

print(
    pd.DataFrame(
        l1_cm,
        index=labels,
        columns=labels
    )
)


print("\n" + "=" * 70)
print("L2 CONFUSION MATRIX")
print("=" * 70)

print(
    pd.DataFrame(
        l2_cm,
        index=labels,
        columns=labels
    )
)


# ---------------------------------------------------------------
# Step 14: Compare L1 vs L2
# ---------------------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression - L1",
        "Logistic Regression - L2"
    ],

    "Accuracy": [
        l1_accuracy,
        l2_accuracy
    ],

    "Weighted_Precision": [
        l1_precision,
        l2_precision
    ],

    "Weighted_Recall": [
        l1_recall,
        l2_recall
    ],

    "Weighted_F1": [
        l1_f1,
        l2_f1
    ],

    "Macro_F1": [
        l1_macro_f1,
        l2_macro_f1
    ]
})


# ---------------------------------------------------------------
# Step 15: Display comparison
# ---------------------------------------------------------------

print("\n" + "=" * 70)
print("L1 vs L2 COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ---------------------------------------------------------------
# Step 16: Determine better model
# ---------------------------------------------------------------

best_model_row = comparison.loc[
    comparison["Macro_F1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Based on Macro F1: "
    f"{best_model_row['Model']}"
)

print(
    f"Macro F1: "
    f"{best_model_row['Macro_F1']:.4f}"
)


# ---------------------------------------------------------------
# Step 17: Save models
# ---------------------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(
    l1_model,
    L1_MODEL_FILE
)

joblib.dump(
    l2_model,
    L2_MODEL_FILE
)

comparison.to_csv(
    COMPARISON_FILE,
    index=False
)


# ---------------------------------------------------------------
# Final output
# ---------------------------------------------------------------

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"L1 model : {L1_MODEL_FILE}")
print(f"L2 model : {L2_MODEL_FILE}")
print(f"Comparison: {COMPARISON_FILE}")

print("\nL1 vs L2 validation completed.")