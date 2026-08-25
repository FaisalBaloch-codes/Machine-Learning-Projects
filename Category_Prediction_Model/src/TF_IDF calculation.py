"""
News Classification - Train/Test Split + TF-IDF
================================================

Workflow:
1. Load the cleaned news dataset.
2. Remove rows with missing Category or Title.
3. Separate Title (X) and Category (y).
4. Split the data into 80% training and 20% testing.
5. Use stratified splitting to preserve category proportions.
6. Fit TF-IDF ONLY on the training titles.
7. Transform the testing titles using the same TF-IDF vectorizer.
8. Use min_df and max_df to filter the vocabulary.
9. Save the split datasets and TF-IDF matrices for the next step.

Important:
- TF-IDF is NEVER fitted on the complete dataset.
- The test set remains unseen during TF-IDF fitting.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

INPUT_FILE = "data/News_Classification_Labeling.csv"

TRAIN_OUTPUT_FILE = "data/train_data.csv"
TEST_OUTPUT_FILE = "data/test_data.csv"

TRAIN_TFIDF_FILE = "data/train_tfidf.csv"
TEST_TFIDF_FILE = "data/test_tfidf.csv"

# Vocabulary filtering
MIN_DF = 2
MAX_DF = 0.95

# Test size = 20%
TEST_SIZE = 0.20

# Makes the split reproducible
RANDOM_STATE = 42


# ------------------------------------------------------------------
# Step 1: Load and clean data
# ------------------------------------------------------------------

def load_and_clean_data(filepath):
    """Load the cleaned dataset and prepare Title and Category."""

    df = pd.read_csv(filepath)

    # Remove rows without a target category
    df = df.dropna(subset=["Category"])

    # Remove rows without a title
    df = df.dropna(subset=["Title"])

    # Convert Title to string
    df["Title"] = df["Title"].astype(str)

    # Remove unnecessary whitespace
    df["Title"] = df["Title"].str.strip()

    # Remove empty titles
    df = df[df["Title"] != ""]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ------------------------------------------------------------------
# Step 2: Split into 80% training and 20% testing
# ------------------------------------------------------------------

def split_data(df):
    """
    Split the dataset into training and testing sets.

    stratify=y ensures that the category proportions are
    approximately preserved in both datasets.
    """

    X = df["Title"]
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------
# Step 3: Save the train/test datasets
# ------------------------------------------------------------------

def save_split_data(X_train, X_test, y_train, y_test):
    """Save the training and testing data for inspection."""

    train_df = pd.DataFrame({
        "Title": X_train.values,
        "Category": y_train.values
    })

    test_df = pd.DataFrame({
        "Title": X_test.values,
        "Category": y_test.values
    })

    train_df.to_csv(TRAIN_OUTPUT_FILE, index=False)
    test_df.to_csv(TEST_OUTPUT_FILE, index=False)

    return train_df, test_df


# ------------------------------------------------------------------
# Step 4: Fit TF-IDF ONLY on training data
# ------------------------------------------------------------------

def calculate_tfidf(X_train, X_test):
    """
    Fit TF-IDF on training data and transform both training
    and testing data.

    min_df:
        Ignore words appearing in fewer than MIN_DF documents.

    max_df:
        Ignore words appearing in more than MAX_DF proportion
        of documents.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=MIN_DF,
        max_df=MAX_DF
    )

    # IMPORTANT:
    # fit_transform is used ONLY on training data.
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # IMPORTANT:
    # Test data is only transformed.
    X_test_tfidf = vectorizer.transform(X_test)

    return vectorizer, X_train_tfidf, X_test_tfidf


# ------------------------------------------------------------------
# Step 5: Save TF-IDF matrices
# ------------------------------------------------------------------

def save_tfidf_data(
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test,
    vectorizer
):
    """Convert TF-IDF matrices to DataFrames and save them."""

    feature_names = vectorizer.get_feature_names_out()

    tfidf_columns = [
        f"tfidf_{word}"
        for word in feature_names
    ]

    train_tfidf_df = pd.DataFrame(
        X_train_tfidf.toarray(),
        columns=tfidf_columns
    )

    test_tfidf_df = pd.DataFrame(
        X_test_tfidf.toarray(),
        columns=tfidf_columns
    )

    # Add target category
    train_tfidf_df["Category"] = y_train.values
    test_tfidf_df["Category"] = y_test.values

    train_tfidf_df.to_csv(
        TRAIN_TFIDF_FILE,
        index=False
    )

    test_tfidf_df.to_csv(
        TEST_TFIDF_FILE,
        index=False
    )

    return train_tfidf_df, test_tfidf_df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    print("=" * 60)
    print("NEWS CLASSIFICATION - TF-IDF PREPARATION")
    print("=" * 60)

    # 1. Load data
    df = load_and_clean_data(INPUT_FILE)

    print(f"\nTotal usable rows: {len(df)}")

    # 2. Split 80/20
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n80/20 SPLIT")
    print("-" * 40)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    # 3. Show category distribution
    print("\nTraining category distribution:")
    print(y_train.value_counts())

    print("\nTesting category distribution:")
    print(y_test.value_counts())

    # 4. Save train/test data
    train_df, test_df = save_split_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # 5. TF-IDF
    print("\nCalculating TF-IDF...")
    print(f"min_df = {MIN_DF}")
    print(f"max_df = {MAX_DF}")

    vectorizer, X_train_tfidf, X_test_tfidf = calculate_tfidf(
        X_train,
        X_test
    )

    # 6. Display vocabulary size
    print("\nTF-IDF RESULTS")
    print("-" * 40)
    print(f"Number of TF-IDF features: {X_train_tfidf.shape[1]}")
    print(f"Training TF-IDF shape: {X_train_tfidf.shape}")
    print(f"Testing TF-IDF shape : {X_test_tfidf.shape}")

    # 7. Save TF-IDF data
    save_tfidf_data(
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
        vectorizer
    )

    print("\nFiles created:")
    print(f"  {TRAIN_OUTPUT_FILE}")
    print(f"  {TEST_OUTPUT_FILE}")
    print(f"  {TRAIN_TFIDF_FILE}")
    print(f"  {TEST_TFIDF_FILE}")

    print("\nTF-IDF preparation completed.")


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()