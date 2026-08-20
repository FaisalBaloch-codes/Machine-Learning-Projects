"""
cleaning/preprocess_for_nlp.py

Runs AFTER clean_articles.py. Takes the cleaned CSV (data/8000_articles_clean.csv)
and prepares the text for model training.

Since the model type/task hasn't been decided yet, this script keeps
BOTH versions of the text so you don't have to redo preprocessing later:

  - "content"            -> untouched, human-readable (already cleaned by
                             clean_articles.py). Use this if you end up
                             using a transformer/BERT-style model - those
                             expect natural text, NOT stopword-stripped,
                             stemmed text.
  - "content_processed"  -> lowercased, punctuation/numbers removed,
                             stopwords removed, lemmatized, extra
                             whitespace collapsed. Use this if you end up
                             using classic ML (Naive Bayes, SVM, Logistic
                             Regression, TF-IDF, etc.) or a from-scratch
                             LSTM/CNN with a bag-of-words / TF-IDF input.

Same "title_processed" is also created from "title" in case you want to
use headlines as a feature too.

Rows where "content" is empty (the url+source-only rows kept by
clean_articles.py) are DROPPED here, since they have no text a model
can learn from. Everything else is preserved.

Run with:
    python cleaning/preprocess_for_nlp.py
"""

import os
import re
import string

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# --- Configuration ---

CLEAN_CSV_PATH = "data/8000_articles_clean.csv"
NLP_READY_CSV_PATH = "data/8000_articles_nlp_ready.csv"

TEXT_COLUMNS_TO_PROCESS = ["title", "content"]


# =========================================================
# 0. ENSURE NLTK DATA IS AVAILABLE
# =========================================================

def ensure_nltk_data():
    """Downloads required NLTK resources quietly if not already present."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# =========================================================
# 1. LOAD CLEANED DATA
# =========================================================

def load_clean_data(filepath=CLEAN_CSV_PATH):
    print("=== 1. LOAD CLEANED DATA ===")
    df = pd.read_csv(filepath, keep_default_na=False)  # keep_default_na=False so "" stays "" not NaN
    print(f"Rows loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")
    return df


# =========================================================
# 2. DROP ROWS WITH NO CONTENT
# =========================================================

def drop_empty_content(df):
    """
    Drops rows where 'content' is empty - a model can't learn from
    those. (These are the url+source-only rows that clean_articles.py
    intentionally kept for record-keeping.)
    """
    print("=== 2. DROP ROWS WITH NO CONTENT ===")

    before = len(df)
    df = df[df["content"].str.strip() != ""].copy()
    dropped = before - len(df)

    print(f"Rows dropped (empty content): {dropped}")
    print(f"Rows remaining: {len(df)}\n")

    return df, dropped


# =========================================================
# 3. NLP TEXT PREPROCESSING (for classic ML)
# =========================================================

STOPWORDS = None
LEMMATIZER = None


def _init_nlp_tools():
    global STOPWORDS, LEMMATIZER
    if STOPWORDS is None:
        STOPWORDS = set(stopwords.words("english"))
    if LEMMATIZER is None:
        LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text):
    """
    Full classic-NLP preprocessing pipeline for one string:
      1. Lowercase
      2. Remove URLs (leftover ones inside article text, if any)
      3. Remove punctuation
      4. Remove digits
      5. Tokenize
      6. Remove stopwords
      7. Lemmatize
      8. Rejoin into a single space-separated string
    """
    _init_nlp_tools()

    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\d+", " ", text)                        # remove digits
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def add_processed_columns(df, columns=TEXT_COLUMNS_TO_PROCESS):
    print("=== 3. NLP TEXT PREPROCESSING (classic-ML version) ===")
    print("(lowercase, remove punctuation/digits/URLs, remove stopwords, lemmatize)")

    for col in columns:
        new_col = f"{col}_processed"
        df[new_col] = df[col].apply(preprocess_text)
        print(f"Created column: {new_col}")

    print()
    return df


# =========================================================
# 4. DROP ROWS WHERE PROCESSED CONTENT BECAME EMPTY
# =========================================================

def drop_empty_after_processing(df):
    """
    Some articles may consist entirely of stopwords/numbers/punctuation
    (rare, but possible) and become empty after processing. Those can't
    be used for classic ML (empty TF-IDF vector), so flag/drop them.
    Original 'content' is untouched, so nothing is lost for transformer use -
    only rows genuinely unusable for BOTH paths are removed here.
    """
    print("=== 4. DROP ROWS WHERE PROCESSED CONTENT IS EMPTY ===")

    before = len(df)
    df = df[df["content_processed"].str.strip() != ""].copy()
    dropped = before - len(df)

    print(f"Rows dropped (empty after NLP processing): {dropped}")
    print(f"Rows remaining: {len(df)}\n")

    return df, dropped


# =========================================================
# 5. VALIDATE
# =========================================================

def validate_data(df):
    print("=== 5. VALIDATE ===")

    stats = {
        "total_rows": len(df),
        "empty_content_processed": (df["content_processed"] == "").sum(),
        "avg_content_word_count": df["content_processed"].str.split().apply(len).mean(),
        "avg_title_word_count": df["title_processed"].str.split().apply(len).mean(),
    }

    for key, value in stats.items():
        print(f"{key}: {value}")
    print()

    return stats


# =========================================================
# 6. SAVE
# =========================================================

def save_nlp_ready_data(df, filepath=NLP_READY_CSV_PATH):
    print("=== 6. SAVE NLP-READY DATA ===")

    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}\n")


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_nltk_data()

    df = load_clean_data()
    raw_rows = len(df)

    df, dropped_empty_content = drop_empty_content(df)
    df = add_processed_columns(df)
    df, dropped_after_processing = drop_empty_after_processing(df)
    stats = validate_data(df)
    save_nlp_ready_data(df)

    print("=== NLP PREPROCESSING COMPLETE ===")
    print(f"Starting rows: {raw_rows}")
    print(f"Dropped (no content): {dropped_empty_content}")
    print(f"Dropped (empty after NLP cleaning): {dropped_after_processing}")
    print(f"Final rows: {stats['total_rows']}")
    print(f"Output file: {NLP_READY_CSV_PATH}")
    print(f"\nColumns available: {list(df.columns)}")
    print("-> Use 'content' / 'title' for transformer models")
    print("-> Use 'content_processed' / 'title_processed' for classic ML (TF-IDF, etc.)")


if __name__ == "__main__":
    main()