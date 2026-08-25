"""
News Classification - Logistic Regression
==========================================

This script:
1. Loads the training and testing TF-IDF datasets.
2. Separates TF-IDF features (X) from Category (y).
3. Trains a Logistic Regression classifier.
4. Uses L2 regularization.
5. Makes predictions on the unseen test set.
6. Saves the trained model.
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
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------------
# File paths
# ---------------------------------------------------------------

TRAIN_FILE = "data/train_tfidf.csv"
TEST_FILE = "data/test_tfidf.csv"

MODEL_FILE = "models/logistic_regression_model.pkl"


# ---------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------

print("=" * 60)
print("LOGISTIC REGRESSION - NEWS CATEGORY CLASSIFICATION")
print("=" * 60)

print("\nLoading training and testing data...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training rows: {len(train_df)}")
print(f"Testing rows : {len(test_df)}")


# ---------------------------------------------------------------
# Step 2: Separate features and target
# ---------------------------------------------------------------

# Category is the target variable.
# Everything beginning with "tfidf_" is an input feature.

feature_columns = [
    column
    for column in train_df.columns
    if column.startswith("tfidf_")
]

X_train = train_df[feature_columns]
y_train = train_df["Category"]

X_test = test_df[feature_columns]
y_test = test_df["Category"]

print(f"\nNumber of TF-IDF features: {len(feature_columns)}")


# ---------------------------------------------------------------
# Step 3: Create Logistic Regression model
# ---------------------------------------------------------------

print("\nCreating Logistic Regression model...")

model = LogisticRegression(
    penalty="l2",
    max_iter=1000,
    random_state=42
)


# ---------------------------------------------------------------
# Step 4: Train the model
# ---------------------------------------------------------------

print("Training model...")

model.fit(X_train, y_train)

print("Training completed.")


# ---------------------------------------------------------------
# Step 5: Make predictions
# ---------------------------------------------------------------

print("\nMaking predictions on test data...")

y_pred = model.predict(X_test)


# ---------------------------------------------------------------
# Step 6: Calculate accuracy
# ---------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ---------------------------------------------------------------
# Step 7: Classification report
# ---------------------------------------------------------------

print("\nClassification Report:")
print("-" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ---------------------------------------------------------------
# Step 8: Confusion matrix
# ---------------------------------------------------------------

print("\nConfusion Matrix:")
print("-" * 60)

labels = model.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)


# ---------------------------------------------------------------
# Step 9: Save model
# ---------------------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_FILE)

print("\nModel saved to:")
print(MODEL_FILE)

print("\nLogistic Regression process completed.")