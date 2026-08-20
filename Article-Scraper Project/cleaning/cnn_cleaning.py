"""
cleaning/cnn_cleaner.py

Cleans the raw CNN scrape data using pandas and saves the result to
data/cnn_articles.csv. The original raw CSV is never modified - this
script only reads it.

Run with:
    python cleaning/cnn_cleaner.py
"""

import os
import pandas as pd


# --- Configuration ---

RAW_CSV_PATH = "data/cnn_raw_articles.csv"
CLEAN_CSV_PATH = "data/cnn_articles.csv"

# Final column set, in the exact required order
FINAL_COLUMNS = ["url", "title", "description", "author", "publish_date", "content"]

# Columns that get "single line" text cleaning (collapse ALL whitespace,
# including newlines, down to single spaces)
SINGLE_LINE_TEXT_COLUMNS = ["title", "description", "author"]


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_raw_data(filepath=RAW_CSV_PATH):
    """
    Reads the raw scraped CSV and prints a quick inspection summary.

    Tries utf-8 first. Falls back to cp1252 (common when a CSV is
    written on Windows without explicitly specifying encoding="utf-8"),
    then latin-1 as a last resort, so a text-encoding mismatch doesn't
    crash the script.
    """
    encodings_to_try = ["utf-8", "cp1252", "latin-1"]

    df = None
    used_encoding = None
    last_error = None

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError as error:
            last_error = error
            continue

    if df is None:
        raise last_error

    print("=== 1. LOAD DATA ===")
    if used_encoding != "utf-8":
        print(f"Note: file was not valid UTF-8 - read successfully using '{used_encoding}' instead.")
    print(f"Initial rows: {len(df)}")
    print(f"Initial columns: {list(df.columns)}\n")

    return df


# =========================================================
# 2. KEEP ONLY THE 6 REQUIRED COLUMNS
# =========================================================

def select_required_columns(df, columns=FINAL_COLUMNS):
    """
    Keeps only url/title/description/author/publish_date/content, in
    that exact order. If one is unexpectedly missing from the raw file,
    it's added as empty rather than crashing.
    """
    print("=== 2. KEEP REQUIRED COLUMNS ===")

    for col in columns:
        if col not in df.columns:
            print(f"Warning: expected column '{col}' was not found in the raw data - adding it as empty.")
            df[col] = ""

    df = df[columns].copy()

    print(f"Columns kept: {list(df.columns)}\n")
    return df


# =========================================================
# 3. REMOVE COMPLETELY EMPTY ROWS
# =========================================================

def remove_empty_rows(df):
    """
    Removes rows where every one of the 6 columns is empty/NaN
    (i.e. the row carries no information at all).
    """
    print("=== 3. REMOVE COMPLETELY EMPTY ROWS ===")

    # fillna("") BEFORE astype(str) - this reliably turns real missing
    # values into an empty string regardless of pandas version/dtype
    # quirks (some pandas string dtypes don't convert NaN to the text
    # "nan" via astype(str) the way older/object dtypes did).
    is_blank = df.apply(lambda col: col.fillna("").astype(str).str.strip()) == ""
    all_blank_mask = is_blank.all(axis=1)

    removed = all_blank_mask.sum()
    df = df[~all_blank_mask].copy()

    print(f"Completely empty rows removed: {removed}")
    print(f"Rows remaining: {len(df)}\n")

    return df, removed


# =========================================================
# 4. REMOVE DUPLICATE URLS
# =========================================================

def remove_duplicate_urls(df):
    """Removes duplicate articles based on the url column, keeping the first occurrence."""
    print("=== 5. REMOVE DUPLICATE URLS ===")

    duplicate_count = df.duplicated(subset="url").sum()
    df = df.drop_duplicates(subset="url", keep="first")

    print(f"Duplicate URLs removed: {duplicate_count}")
    print(f"Rows remaining: {len(df)}\n")

    return df, duplicate_count


# =========================================================
# 5. CLEAN TEXT FIELDS
# =========================================================

def clean_single_line_text(series):
    """
    For title/description/author: strip HTML tags, collapse ALL
    whitespace (including newlines) into single spaces, trim ends, and
    treat whitespace-only values as empty.
    """
    series = series.fillna("").astype(str)
    series = series.str.replace(r"<[^>]+>", "", regex=True)   # strip leftover HTML tags
    series = series.str.replace(r"\s+", " ", regex=True)      # collapse all whitespace
    series = series.str.strip()
    return series


def clean_content_text(series):
    """
    For the article body: strip HTML tags, normalize newline styles,
    collapse repeated spaces/tabs (but keep newlines so paragraphs stay
    separated), and collapse 3+ blank lines down to a single blank line.
    Content itself is never shortened, summarized, or rewritten.
    """
    series = series.fillna("").astype(str)
    series = series.str.replace(r"<[^>]+>", "", regex=True)          # strip leftover HTML tags
    series = series.str.replace(r"\r\n|\r", "\n", regex=True)        # normalize newlines to \n
    series = series.str.replace(r"[ \t]+", " ", regex=True)          # collapse spaces/tabs, keep \n
    series = series.str.replace(r"\n{3,}", "\n\n", regex=True)       # collapse excess blank lines
    series = series.str.strip()
    return series


def clean_text_fields(df):
    """Applies the appropriate cleaning to title/description/author/content."""
    print("=== 6. CLEAN TEXT FIELDS ===")

    for col in SINGLE_LINE_TEXT_COLUMNS:
        df[col] = clean_single_line_text(df[col])

    df["content"] = clean_content_text(df["content"])

    print(f"Cleaned columns: {SINGLE_LINE_TEXT_COLUMNS + ['content']}\n")
    return df


# =========================================================
# 6. CLEAN PUBLISH DATE
# =========================================================

def clean_publish_date(df):
    """
    Converts publish_date to a consistent 'YYYY-MM-DD' string using
    pandas.to_datetime with errors="coerce", so invalid/unparseable
    dates become missing instead of crashing or being guessed at.
    """
    print("=== 7. CLEAN PUBLISH DATE ===")

    parsed_dates = pd.to_datetime(df["publish_date"], errors="coerce", utc=True)

    converted_count = parsed_dates.notna().sum()
    missing_count = parsed_dates.isna().sum()

    print(f"Dates successfully converted: {converted_count}")
    print(f"Dates missing/invalid: {missing_count}")

    formatted = parsed_dates.dt.strftime("%Y-%m-%d")
    df["publish_date"] = formatted  # stays NaN where parsing failed - not invented

    print()
    return df


# =========================================================
# 7. REMOVE ROWS WITH MISSING URL
# =========================================================

def remove_missing_url_rows(df):
    """Removes rows where the url is empty/NaN - a row is useless without it."""
    print("=== 4. REMOVE ROWS WITH MISSING URL ===")

    before = len(df)
    url_stripped = df["url"].fillna("").astype(str).str.strip()
    df = df[url_stripped != ""].copy()
    removed = before - len(df)

    print(f"Rows removed (missing url): {removed}")
    print(f"Rows remaining: {len(df)}\n")

    return df, removed


# =========================================================
# 8. REMOVE ROWS WHERE TITLE AND CONTENT ARE BOTH MISSING
# =========================================================

def remove_missing_title_and_content_rows(df):
    """
    Removes rows where BOTH title and content are empty - such a row
    has no usable article information even if other fields exist.
    """
    print("=== 8. REMOVE ROWS MISSING BOTH TITLE AND CONTENT ===")

    before = len(df)
    title_empty = df["title"].fillna("").astype(str).str.strip() == ""
    content_empty = df["content"].fillna("").astype(str).str.strip() == ""

    both_missing_mask = title_empty & content_empty
    removed = both_missing_mask.sum()
    df = df[~both_missing_mask].copy()

    print(f"Rows removed (title AND content both missing): {removed}")
    print(f"Rows remaining: {len(df)}\n")

    return df, removed


# =========================================================
# 9-10. VALIDATION STATS (missing-value counts, no fake data inserted)
# =========================================================

def compute_missing_value_stats(df):
    """
    Computes how many rows have each field missing/empty AFTER cleaning.
    Nothing is filled in with fake data anywhere in this script - a
    missing value simply stays as an empty string (text) or NaN (date).
    """
    print("=== 9. MISSING VALUE SUMMARY ===")

    def count_empty(col):
        return (df[col].fillna("").astype(str).str.strip() == "").sum()

    stats = {
        "missing_titles": count_empty("title"),
        "missing_descriptions": count_empty("description"),
        "missing_authors": count_empty("author"),
        "missing_publish_dates": df["publish_date"].isna().sum(),
        "missing_content": count_empty("content"),
    }

    for key, value in stats.items():
        print(f"{key}: {value}")
    print()

    return stats


# =========================================================
# 11. SAVE CLEAN DATA
# =========================================================

def save_clean_data(df, filepath=CLEAN_CSV_PATH):
    """Saves the cleaned DataFrame to CSV, creating the output folder if needed."""
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(filepath, index=False)


# =========================================================
# 12. VERIFY OUTPUT COLUMNS
# =========================================================

def verify_output_columns(df, expected=FINAL_COLUMNS):
    """Confirms the final DataFrame has exactly the expected columns, in order."""
    actual = list(df.columns)
    if actual != expected:
        print(f"WARNING: output columns {actual} do not match the expected {expected}!")
        return False
    return True


# =========================================================
# MAIN
# =========================================================

def main():
    df = load_raw_data()
    original_rows = len(df)

    df = select_required_columns(df)
    df, empty_rows_removed = remove_empty_rows(df)
    df, missing_url_removed = remove_missing_url_rows(df)
    df, duplicate_urls_removed = remove_duplicate_urls(df)
    df = clean_text_fields(df)
    df = clean_publish_date(df)
    df, missing_title_content_removed = remove_missing_title_and_content_rows(df)

    stats = compute_missing_value_stats(df)

    total_removed = original_rows - len(df)

    save_clean_data(df)
    columns_ok = verify_output_columns(df)

    print("=== CNN CLEANING COMPLETE ===")
    print(f"Original number of rows: {original_rows}")
    print(f"Rows removed: {total_removed}")
    print(f"  - Completely empty rows: {empty_rows_removed}")
    print(f"  - Duplicate URLs: {duplicate_urls_removed}")
    print(f"  - Missing URL: {missing_url_removed}")
    print(f"  - Missing title AND content: {missing_title_content_removed}")
    print(f"Final number of rows: {len(df)}")
    print(f"Missing titles: {stats['missing_titles']}")
    print(f"Missing descriptions: {stats['missing_descriptions']}")
    print(f"Missing authors: {stats['missing_authors']}")
    print(f"Missing publish dates: {stats['missing_publish_dates']}")
    print(f"Missing content values: {stats['missing_content']}")

    print(f"\nOutput file: {CLEAN_CSV_PATH}")
    print(f"Final row count: {len(df)}")
    print(f"Final column names: {list(df.columns)}")

    if columns_ok:
        print(f"Column verification passed: columns match {FINAL_COLUMNS} exactly.")


if __name__ == "__main__":
    main()