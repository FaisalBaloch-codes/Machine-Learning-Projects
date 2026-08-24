"""
News Classification - Feature Engineering & TF-IDF Vectorization
==================================================================

This script:
1. Loads and cleans the raw news dataset.
2. Extracts 7 manually engineered numerical features from the `Title` column.
3. Computes Top-500 TF-IDF features from the `Title` column, with columns
   dynamically named after the actual words (e.g. `tfidf_bitcoin`).
4. Concatenates everything into a single DataFrame in a fixed column order
   and exports it to `News_Classification_Processed.csv`.

Run with:
    python process_news_data.py
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import re                                              # regex-based text parsing (digits, punctuation, etc.)
import pandas as pd                                    # data loading / manipulation
from textblob import TextBlob                          # sentiment polarity scoring
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF vectorization


# ------------------------------------------------------------------
# Config / constants
# ------------------------------------------------------------------
INPUT_FILE = "News Classification Labeling.csv"
OUTPUT_FILE = "News_Classification_Processed.csv"
MAX_TFIDF_FEATURES = 500

# A small, illustrative list of well-known geopolitical / regional terms.
# Extend this list as needed for your specific dataset/domain.
LOCATION_TERMS = {
    "us", "usa", "u.s.", "u.s.a.", "united states", "america",
    "china", "russia", "india", "uk", "britain", "united kingdom",
    "europe", "eu", "france", "germany", "japan", "canada",
    "australia", "africa", "asia", "middle east", "ukraine",
    "israel", "gaza", "iran", "iraq", "korea", "mexico", "brazil",
    "new york", "washington", "california", "texas", "london",
    "beijing", "moscow", "paris", "berlin", "tokyo", "wall street",
}


# ------------------------------------------------------------------
# Step 1: Data Ingestion & Cleaning
# ------------------------------------------------------------------
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV and perform basic cleaning."""
    df = pd.read_csv(filepath)

    # Drop rows where the target label 'Category' is missing - these rows
    # are unusable for a supervised classification task.
    df = df.dropna(subset=["Category"])

    # Replace missing titles with an empty string so downstream string
    # operations (len, regex, TF-IDF, etc.) don't break on NaN.
    df["Title"] = df["Title"].fillna("")

    # Reset index after dropping rows to keep things tidy.
    df = df.reset_index(drop=True)

    return df


# ------------------------------------------------------------------
# Step 2: Manual Feature Engineering (7 features)
# ------------------------------------------------------------------
def extract_manual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 7 numerical features derived from the Title column."""

    titles = df["Title"].astype(str)

    # 1. title_len: total character length of the title
    df["title_len"] = titles.apply(len)

    # 2. num_digits_count: count of numeric digits (0-9) in the title
    df["num_digits_count"] = titles.apply(lambda t: sum(c.isdigit() for c in t))

    # 3. question_mark_count: frequency count of '?' characters
    df["question_mark_count"] = titles.apply(lambda t: t.count("?"))

    # 4. punctuation_count: total count of punctuation characters
    #    (anything that's not alphanumeric or whitespace)
    punctuation_pattern = re.compile(r"[^\w\s]")
    df["punctuation_count"] = titles.apply(lambda t: len(punctuation_pattern.findall(t)))

    # 5. sentiment_score: polarity score from TextBlob, range [-1.0, 1.0]
    df["sentiment_score"] = titles.apply(lambda t: TextBlob(t).sentiment.polarity)

    # 6. poi_count: count of likely "Points of Interest" - proxied here as
    #    the number of capitalized words (excluding the first word of the
    #    title, which is capitalized regardless due to normal sentence case).
    #    This acts as a rough proxy for named entities / stock tickers /
    #    proper nouns (e.g. "Tesla", "NASA", "AAPL").
    def count_capitalized_terms(title: str) -> int:
        words = title.split()
        count = 0
        for i, w in enumerate(words):
            cleaned = w.strip(".,!?;:'\"()[]")
            if not cleaned:
                continue
            # Skip the very first word - its capitalization is just
            # standard sentence casing, not necessarily an entity.
            if i == 0:
                continue
            # A capitalized alphabetic word (e.g. "Tesla") or an
            # all-caps ticker-like token (e.g. "AAPL", "NASA").
            if cleaned[0].isupper() and cleaned.isalpha():
                count += 1
        return count

    df["poi_count"] = titles.apply(count_capitalized_terms)

    # 7. has_location: binary flag - 1 if any known region/geopolitical
    #    term appears in the title (case-insensitive), else 0.
    def detect_location(title: str) -> int:
        lower_title = title.lower()
        for term in LOCATION_TERMS:
            # Use word-boundary matching so short terms like "us" don't
            # falsely match inside other words (e.g. "bonus").
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, lower_title):
                return 1
        return 0

    df["has_location"] = titles.apply(detect_location)

    return df


# ------------------------------------------------------------------
# Step 3: Top-500 TF-IDF Vectorization (Dynamic Column Names)
# ------------------------------------------------------------------
def compute_tfidf_features(df: pd.DataFrame, max_features: int = MAX_TFIDF_FEATURES) -> pd.DataFrame:
    """Fit a TF-IDF vectorizer on Title and return a DataFrame of
    tfidf_<word> columns aligned to df's index."""

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)

    # Fit on the Title column and transform it into a sparse TF-IDF matrix.
    tfidf_matrix = vectorizer.fit_transform(df["Title"].astype(str))

    # Retrieve the exact vocabulary words chosen by the vectorizer
    # (already sorted alphabetically by sklearn).
    feature_names = vectorizer.get_feature_names_out()

    # Dynamically name each column using the real word, e.g. "tfidf_bitcoin".
    tfidf_columns = [f"tfidf_{word}" for word in feature_names]

    # Convert the sparse matrix to a dense DataFrame with the same index
    # as the original df, so concatenation aligns correctly.
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_columns,
        index=df.index,
    )

    return tfidf_df


# ------------------------------------------------------------------
# Step 4: Assemble Final DataFrame & Export
# ------------------------------------------------------------------
def build_final_dataframe(df: pd.DataFrame, tfidf_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate all pieces into the final column order:
    Title, Category, 7 manual features, then the 500 TF-IDF columns."""

    manual_feature_cols = [
        "title_len",
        "num_digits_count",
        "question_mark_count",
        "punctuation_count",
        "sentiment_score",
        "poi_count",
        "has_location",
    ]

    base_df = df[["Title", "Category"] + manual_feature_cols]

    final_df = pd.concat([base_df, tfidf_df], axis=1)

    return final_df


# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
def main():
    # Step 1: load + clean
    df = load_and_clean_data(INPUT_FILE)

    # Step 2: manual engineered features
    df = extract_manual_features(df)

    # Step 3: TF-IDF features
    tfidf_df = compute_tfidf_features(df, max_features=MAX_TFIDF_FEATURES)

    # Step 4: combine + export
    final_df = build_final_dataframe(df, tfidf_df)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed {len(final_df)} rows.")
    print(f"Final shape: {final_df.shape}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()