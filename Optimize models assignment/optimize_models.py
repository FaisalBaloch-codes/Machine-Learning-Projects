"""
News Classification - Hyperparameter Optimization
==================================================
Optimizes Decision Tree and Random Forest classifiers using
GridSearchCV + K-Fold Cross Validation.

Pipeline:
1. Load dataset (same as tfidf_calculation.py)
2. Split 80/20 (same random_state=42, stratified)
3. Fit TF-IDF on train only, transform test (same min_df/max_df)
4. For each model:
   - Define hyperparameter grid
   - Run GridSearchCV with 5-fold cross-validation
   - Retrain with best parameters
   - Evaluate on the held-out test set
   - Save results to a separate Excel file
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------------------------------
# Configuration (matches your original tfidf_calculation.py)
# ------------------------------------------------------------------
INPUT_FILE = "News__Classification_Labeling.csv"

MIN_DF = 2
MAX_DF = 0.95
TEST_SIZE = 0.20
RANDOM_STATE = 42

DT_OUTPUT_FILE = "optimized_decisiontree_results.xlsx"
RF_OUTPUT_FILE = "optimized_randomforest_results.xlsx"


# ------------------------------------------------------------------
# Step 1: Load and clean data
# ------------------------------------------------------------------
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["Category"])
    df = df.dropna(subset=["Title"])
    df["Title"] = df["Title"].astype(str).str.strip()
    df = df[df["Title"] != ""]
    df = df.reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# Step 2: Split 80/20 (stratified, same seed as before)
# ------------------------------------------------------------------
def split_data(df):
    X = df["Title"]
    y = df["Category"]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


# ------------------------------------------------------------------
# Step 3: TF-IDF (fit on train only, same min_df/max_df as before)
# ------------------------------------------------------------------
def calculate_tfidf(X_train, X_test):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=MIN_DF,
        max_df=MAX_DF
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return vectorizer, X_train_tfidf, X_test_tfidf


# ------------------------------------------------------------------
# Step 4: Optimize a model with GridSearchCV + K-Fold, then save results
# ------------------------------------------------------------------
def optimize_and_save(
    model, param_grid, model_name, output_file,
    X_train, X_test, y_train, y_test, titles_test,
    n_features, total_records
):
    print("\n" + "=" * 60)
    print(f"OPTIMIZING: {model_name}")
    print("=" * 60)

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kfold,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"Best Params: {best_params}")
    print(f"Best CV Accuracy: {best_cv_score*100:.2f}%")

    # Evaluate best model on the held-out test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    print(f"Test Accuracy (optimized): {test_accuracy*100:.2f}%")

    # ---------------- Project Summary sheet ----------------
    summary_df = pd.DataFrame({
        "Item": [
            "Dataset", "Total Records", "TF-IDF Features", "Target Variable",
            "Number of Categories", "Train/Test Split", "Training Records",
            "Testing Records", "Classification Algorithm",
            "Optimization Method", "Cross-Validation Method",
            "Best Hyperparameters", "Best CV Accuracy",
            "Test Accuracy (After Optimization)"
        ],
        "Result": [
            "News Classification", total_records, n_features, "Category",
            y_train.nunique(), "80% / 20%", X_train.shape[0],
            X_test.shape[0], model_name,
            "GridSearchCV", "5-Fold Cross Validation",
            str(best_params), f"{best_cv_score*100:.2f}%",
            f"{test_accuracy*100:.2f}%"
        ]
    })

    # ---------------- Predictions sheet ----------------
    predictions_df = pd.DataFrame({
        "Title": titles_test.values,
        "Actual Category": y_test.values,
        "Predicted Category": y_pred
    })

    # ---------------- Save workbook ----------------
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Project Summary", index=False)
        report_df.to_excel(writer, sheet_name="Classification Report")
        predictions_df.to_excel(writer, sheet_name="Predictions", index=False)

    print(f"Saved -> {output_file}")
    return best_params, best_cv_score, test_accuracy


def main():
    print("Loading data...")
    df = load_and_clean_data(INPUT_FILE)
    total_records = len(df)
    print(f"Usable records: {total_records}")

    print("Splitting 80/20 (stratified)...")
    X_train, X_test, y_train, y_test = split_data(df)

    # Keep the actual titles for the test set (for the Predictions sheet)
    titles_test = X_test.copy()

    print(f"Calculating TF-IDF (min_df={MIN_DF}, max_df={MAX_DF})...")
    vectorizer, X_train_tfidf, X_test_tfidf = calculate_tfidf(X_train, X_test)
    n_features = X_train_tfidf.shape[1]
    print(f"TF-IDF features: {n_features}")

    # ---------------- Decision Tree ----------------
    dt_param_grid = {
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]
    }
    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)

    optimize_and_save(
        dt_model, dt_param_grid,
        "Decision Tree Classifier (Optimized)", DT_OUTPUT_FILE,
        X_train_tfidf, X_test_tfidf, y_train, y_test, titles_test,
        n_features, total_records
    )

    # ---------------- Random Forest ----------------
    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [20, None],
        "min_samples_split": [2, 5]
    }
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)

    optimize_and_save(
        rf_model, rf_param_grid,
        "Random Forest Classifier (Optimized)", RF_OUTPUT_FILE,
        X_train_tfidf, X_test_tfidf, y_train, y_test, titles_test,
        n_features, total_records
    )

    print("\nAll done!")
    print(f"  {DT_OUTPUT_FILE}")
    print(f"  {RF_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
