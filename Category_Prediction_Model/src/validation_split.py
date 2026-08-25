"""
News Classification - Validation Split
=======================================

This script creates a NEW validation experiment without changing
any of the existing baseline files.

IMPORTANT:
The original baseline files are NOT overwritten.

Current baseline:
    train_data.csv
    test_data.csv
    train_tfidf.csv
    test_tfidf.csv

New validation files:
    train_validation_data.csv
    validation_data.csv
    validation_test_data.csv

    train_validation_tfidf.csv
    validation_tfidf.csv
    validation_test_tfidf.csv

Split:
    Original dataset
          |
          +---- 80% Development
          |        |
          |        +---- 90% Training
          |        |
          |        +---- 10% Validation
          |
          +---- 20% Final Test

The final test set remains completely untouched.
"""

# ---------------------------------------------------------------
# Imports
# ---------------------------------------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------
# File paths
# ---------------------------------------------------------------

INPUT_FILE = "data/News_Classification_Labeling.csv"

# NEW files only - existing files will NOT be overwritten
TRAIN_DATA_FILE = "data/train_validation_data.csv"
VALIDATION_DATA_FILE = "data/validation_data.csv"
TEST_DATA_FILE = "data/validation_test_data.csv"

TRAIN_TFIDF_FILE = "data/train_validation_tfidf.csv"
VALIDATION_TFIDF_FILE = "data/validation_tfidf.csv"
TEST_TFIDF_FILE = "data/validation_test_tfidf.csv"


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

MIN_DF = 2
MAX_DF = 0.95

RANDOM_STATE = 42


# ---------------------------------------------------------------
# Step 1: Load and clean data
# ---------------------------------------------------------------

def load_data():

    df = pd.read_csv(INPUT_FILE)

    # Remove rows without Category
    df = df.dropna(subset=["Category"])

    # Remove rows without Title
    df = df.dropna(subset=["Title"])

    # Convert Title to string
    df["Title"] = df["Title"].astype(str).str.strip()

    # Remove empty titles
    df = df[df["Title"] != ""]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------------------
# Step 2: Original 80/20 split
# ---------------------------------------------------------------

def create_original_split(df):

    X = df["Title"]
    y = df["Category"]

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_development, X_test, y_development, y_test


# ---------------------------------------------------------------
# Step 3: 90/10 split of the 80% development data
# ---------------------------------------------------------------

def create_validation_split(
    X_development,
    y_development
):

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_development
    )

    return (
        X_train,
        X_validation,
        y_train,
        y_validation
    )


# ---------------------------------------------------------------
# Step 4: Save new datasets
# ---------------------------------------------------------------

def save_datasets(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test
):

    train_df = pd.DataFrame({
        "Title": X_train.values,
        "Category": y_train.values
    })

    validation_df = pd.DataFrame({
        "Title": X_validation.values,
        "Category": y_validation.values
    })

    test_df = pd.DataFrame({
        "Title": X_test.values,
        "Category": y_test.values
    })

    train_df.to_csv(
        TRAIN_DATA_FILE,
        index=False
    )

    validation_df.to_csv(
        VALIDATION_DATA_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_DATA_FILE,
        index=False
    )

    return train_df, validation_df, test_df


# ---------------------------------------------------------------
# Step 5: TF-IDF
# ---------------------------------------------------------------

def calculate_tfidf(
    X_train,
    X_validation,
    X_test
):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=MIN_DF,
        max_df=MAX_DF
    )

    # IMPORTANT:
    # TF-IDF is fitted ONLY on the new training data.
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Validation is only transformed.
    X_validation_tfidf = vectorizer.transform(X_validation)

    # Final test is only transformed.
    X_test_tfidf = vectorizer.transform(X_test)

    return (
        vectorizer,
        X_train_tfidf,
        X_validation_tfidf,
        X_test_tfidf
    )


# ---------------------------------------------------------------
# Step 6: Save TF-IDF datasets
# ---------------------------------------------------------------

def save_tfidf_datasets(
    X_train_tfidf,
    X_validation_tfidf,
    X_test_tfidf,
    y_train,
    y_validation,
    y_test,
    vectorizer
):

    feature_names = vectorizer.get_feature_names_out()

    tfidf_columns = [
        f"tfidf_{word}"
        for word in feature_names
    ]

    train_tfidf_df = pd.DataFrame(
        X_train_tfidf.astype("float32").toarray(),
        columns=tfidf_columns
    )

    validation_tfidf_df = pd.DataFrame(
        X_validation_tfidf.astype("float32").toarray(),
        columns=tfidf_columns
    )

    test_tfidf_df = pd.DataFrame(
        X_test_tfidf.astype("float32").toarray(),
        columns=tfidf_columns
    )

    # Add target labels
    train_tfidf_df["Category"] = y_train.values
    validation_tfidf_df["Category"] = y_validation.values
    test_tfidf_df["Category"] = y_test.values

    train_tfidf_df.to_csv(
        TRAIN_TFIDF_FILE,
        index=False
    )

    validation_tfidf_df.to_csv(
        VALIDATION_TFIDF_FILE,
        index=False
    )

    test_tfidf_df.to_csv(
        TEST_TFIDF_FILE,
        index=False
    )

    return (
        train_tfidf_df,
        validation_tfidf_df,
        test_tfidf_df
    )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():

    print("=" * 65)
    print("VALIDATION DATASET CREATION")
    print("=" * 65)

    # -----------------------------------------------------------
    # Load data
    # -----------------------------------------------------------

    df = load_data()

    print(f"\nTotal usable rows: {len(df)}")


    # -----------------------------------------------------------
    # Original 80/20 split
    # -----------------------------------------------------------

    (
        X_development,
        X_test,
        y_development,
        y_test
    ) = create_original_split(df)

    print("\nORIGINAL 80/20 SPLIT")
    print("-" * 45)

    print(f"Development data: {len(X_development)}")
    print(f"Final test data : {len(X_test)}")


    # -----------------------------------------------------------
    # 90/10 validation split
    # -----------------------------------------------------------

    (
        X_train,
        X_validation,
        y_train,
        y_validation
    ) = create_validation_split(
        X_development,
        y_development
    )

    print("\nVALIDATION SPLIT")
    print("-" * 45)

    print(f"Training data   : {len(X_train)}")
    print(f"Validation data : {len(X_validation)}")
    print(f"Final test data : {len(X_test)}")


    # -----------------------------------------------------------
    # Verify category distributions
    # -----------------------------------------------------------

    print("\nTRAINING CATEGORY COUNTS")
    print("-" * 45)
    print(y_train.value_counts())

    print("\nVALIDATION CATEGORY COUNTS")
    print("-" * 45)
    print(y_validation.value_counts())

    print("\nFINAL TEST CATEGORY COUNTS")
    print("-" * 45)
    print(y_test.value_counts())


    # -----------------------------------------------------------
    # Save new datasets
    # -----------------------------------------------------------

    save_datasets(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    )

    print("\nNew dataset files created.")


    # -----------------------------------------------------------
    # TF-IDF
    # -----------------------------------------------------------

    print("\nCALCULATING NEW TF-IDF")
    print("-" * 45)

    print(f"min_df = {MIN_DF}")
    print(f"max_df = {MAX_DF}")

    (
        vectorizer,
        X_train_tfidf,
        X_validation_tfidf,
        X_test_tfidf
    ) = calculate_tfidf(
        X_train,
        X_validation,
        X_test
    )

    print(
        f"Number of TF-IDF features: "
        f"{X_train_tfidf.shape[1]}"
    )

    print(
        f"Training TF-IDF shape: "
        f"{X_train_tfidf.shape}"
    )

    print(
        f"Validation TF-IDF shape: "
        f"{X_validation_tfidf.shape}"
    )

    print(
        f"Test TF-IDF shape: "
        f"{X_test_tfidf.shape}"
    )


    # -----------------------------------------------------------
    # Save new TF-IDF datasets
    # -----------------------------------------------------------

    save_tfidf_datasets(
        X_train_tfidf,
        X_validation_tfidf,
        X_test_tfidf,
        y_train,
        y_validation,
        y_test,
        vectorizer
    )


    # -----------------------------------------------------------
    # Final message
    # -----------------------------------------------------------

    print("\n" + "=" * 65)
    print("VALIDATION SETUP COMPLETED")
    print("=" * 65)

    print("\nNEW FILES:")
    print(f"  {TRAIN_DATA_FILE}")
    print(f"  {VALIDATION_DATA_FILE}")
    print(f"  {TEST_DATA_FILE}")
    print(f"  {TRAIN_TFIDF_FILE}")
    print(f"  {VALIDATION_TFIDF_FILE}")
    print(f"  {TEST_TFIDF_FILE}")

    print("\nIMPORTANT:")
    print("Your original baseline files were NOT modified.")


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

if __name__ == "__main__":
    main()