"""
cleaning/bbc_cleaner.py

Cleans the raw BBC scrape data using pandas and saves the result to
data/bbc_articles.csv. The original raw CSV is never modified - this
script only reads it.

Run with:
    python cleaning/bbc_cleaner.py
"""

import os
import pandas as pd


# --- Configuration ---

RAW_CSV_PATH = "data/bbc_raw_articles.csv"
CLEAN_CSV_PATH = "data/bbc_articles.csv"

# Columns we want to keep in the final cleaned dataset
KEEP_COLUMNS = [
    "url", "title", "description", "author", "publish_date",
    "content", "scrape_status", "error_message",
]

# Columns that get "single line" text cleaning (collapse ALL whitespace,
# including newlines, down to single spaces)
SINGLE_LINE_TEXT_COLUMNS = ["title", "description", "author"]


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_raw_data(filepath=RAW_CSV_PATH):
    """
    Reads the raw scraped CSV and prints a quick inspection summary.

    Tries utf-8 first (the expected encoding). If the file actually
    contains cp1252/Windows-1252 bytes (common when a CSV is written on
    Windows without explicitly specifying encoding="utf-8"), falls back
    to that. latin-1 is the final fallback since it can decode any byte
    value, guaranteeing the file will load even if the exact original
    encoding is unknown.
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
    print(f"Initial columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    print("\nMissing values per column:")
    print(df.isna().sum())
    print()

    return df


# =========================================================
# 2. KEEP RELEVANT COLUMNS
# =========================================================

def select_relevant_columns(df, keep_columns=KEEP_COLUMNS):
    """
    Keeps only the columns we care about for the cleaned dataset. If any
    expected column is missing from the raw data (shouldn't normally
    happen), it's added as an empty column instead of crashing, so the
    rest of the pipeline can still run.
    """
    print("=== 2. KEEP RELEVANT COLUMNS ===")

    for col in keep_columns:
        if col not in df.columns:
            print(f"Warning: expected column '{col}' was not found in the raw data - adding it as empty.")
            df[col] = ""

    df = df[keep_columns].copy()

    print(f"Columns kept: {list(df.columns)}\n")
    return df


# =========================================================
# 3. REMOVE UNSUCCESSFUL SCRAPES
# =========================================================

def filter_successful_scrapes(df):
    """Keeps only rows where scrape_status indicates a successful scrape."""
    print("=== 3. REMOVE UNSUCCESSFUL SCRAPES ===")

    # Normalize for comparison only (doesn't change the stored value)
    status_normalized = df["scrape_status"].astype(str).str.strip().str.lower()

    successful_mask = status_normalized == "success"
    successful_count = successful_mask.sum()
    unsuccessful_count = (~successful_mask).sum()

    print(f"Successful rows: {successful_count}")
    print(f"Unsuccessful rows (removed): {unsuccessful_count}")

    df = df[successful_mask].copy()
    print(f"Rows remaining: {len(df)}\n")

    return df, unsuccessful_count


# =========================================================
# 4. REMOVE DUPLICATES
# =========================================================

def remove_duplicates(df):
    """Removes fully duplicated rows, then removes duplicate URLs."""
    print("=== 4. REMOVE DUPLICATES ===")

    full_duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Fully duplicated rows removed: {full_duplicate_count}")

    url_duplicate_count = df.duplicated(subset="url").sum()
    df = df.drop_duplicates(subset="url", keep="first")
    print(f"Duplicate URLs removed: {url_duplicate_count}")

    print(f"Rows remaining: {len(df)}\n")

    return df, full_duplicate_count, url_duplicate_count


# =========================================================
# 5. CLEAN TEXT COLUMNS
# =========================================================

def clean_single_line_text(series):
    """
    For title/description/author: strip HTML tags, collapse ALL
    whitespace (including newlines) into single spaces, and trim ends.
    These fields are expected to read as one line.
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
    Does NOT shorten, summarize, or remove short-but-legitimate sentences.
    """
    series = series.fillna("").astype(str)
    series = series.str.replace(r"<[^>]+>", "", regex=True)          # strip leftover HTML tags
    series = series.str.replace(r"\r\n|\r", "\n", regex=True)        # normalize newlines to \n
    series = series.str.replace(r"[ \t]+", " ", regex=True)          # collapse spaces/tabs, keep \n
    series = series.str.replace(r"\n{3,}", "\n\n", regex=True)       # collapse excess blank lines
    series = series.str.strip()
    return series


def clean_text_columns(df):
    """Applies the appropriate text cleaning to title/description/author/content."""
    print("=== 5. CLEAN TEXT COLUMNS ===")

    for col in SINGLE_LINE_TEXT_COLUMNS:
        df[col] = clean_single_line_text(df[col])

    df["content"] = clean_content_text(df["content"])

    print(f"Cleaned columns: {SINGLE_LINE_TEXT_COLUMNS + ['content']}\n")
    return df


# =========================================================
# 6 & 7. HANDLE MISSING VALUES / CLEAN AUTHOR
# =========================================================

def handle_missing_values(df):
    """
    Fills missing text values with an empty string rather than inventing
    data or dropping rows. Author being empty is expected/legitimate and
    is left as "" rather than flagged as an error.
    """
    print("=== 6. HANDLE MISSING VALUES ===")

    text_columns = SINGLE_LINE_TEXT_COLUMNS + ["content", "error_message"]
    for col in text_columns:
        df[col] = df[col].fillna("")

    print("Missing values after fill (text columns only):")
    print(df[text_columns].isna().sum())
    print()

    return df


# =========================================================
# 8. CLEAN PUBLISH DATE
# =========================================================

def clean_publish_date(df):
    """
    Converts publish_date to a consistent 'YYYY-MM-DD HH:MM:SS' string
    using pandas.to_datetime with errors="coerce" so invalid dates become
    NaT (and then an empty string) instead of crashing or being invented.
    """
    print("=== 8. CLEAN PUBLISH DATE ===")

    parsed_dates = pd.to_datetime(df["publish_date"], errors="coerce", utc=True)

    converted_count = parsed_dates.notna().sum()
    missing_count = parsed_dates.isna().sum()

    print(f"Dates successfully converted: {converted_count}")
    print(f"Dates missing/invalid (NaT): {missing_count}")

    # Format valid dates as plain strings; leave invalid ones as empty string
    formatted = parsed_dates.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["publish_date"] = formatted.fillna("")

    print()
    return df


# =========================================================
# 9. CLEAN URL
# =========================================================

def clean_url(df):
    """Strips whitespace from URLs without altering the URL itself."""
    print("=== 9. CLEAN URL ===")
    df["url"] = df["url"].astype(str).str.strip()
    print("URLs stripped of leading/trailing whitespace.\n")
    return df


# =========================================================
# 10. VALIDATE CONTENT
# =========================================================

def validate_data(df):
    """Computes validation statistics on the cleaned DataFrame."""
    print("=== 10. VALIDATE CONTENT ===")

    stats = {
        "total_rows": len(df),
        "empty_title": (df["title"] == "").sum(),
        "empty_description": (df["description"] == "").sum(),
        "empty_author": (df["author"] == "").sum(),
        "empty_publish_date": (df["publish_date"] == "").sum(),
        "empty_content": (df["content"] == "").sum(),
        "duplicate_urls": df.duplicated(subset="url").sum(),
    }

    content_lengths = df["content"].str.len()
    stats["min_content_length"] = content_lengths.min() if len(df) else 0
    stats["max_content_length"] = content_lengths.max() if len(df) else 0

    for key, value in stats.items():
        print(f"{key}: {value}")
    print()

    return stats


# =========================================================
# 11. SAVE CLEAN DATA
# =========================================================

def save_clean_data(df, filepath=CLEAN_CSV_PATH):
    """Saves the cleaned DataFrame to CSV, creating the output folder if needed."""
    print("=== 11. SAVE CLEAN DATA ===")

    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(filepath, index=False)

    print(f"Saved {len(df)} rows to {filepath}\n")


# =========================================================
# 12. FINAL TERMINAL REPORT
# =========================================================

def print_final_report(raw_row_count, unsuccessful_count, full_dupe_count,
                        url_dupe_count, stats, output_path):
    print("=== BBC CLEANING COMPLETE ===")
    print(f"Raw rows: {raw_row_count}")
    print(f"Successful rows: {raw_row_count - unsuccessful_count}")
    print(f"Rows removed (unsuccessful scrapes): {unsuccessful_count}")
    print(f"Fully duplicated rows removed: {full_dupe_count}")
    print(f"Duplicate URLs removed: {url_dupe_count}")
    print(f"Final rows: {stats['total_rows']}")
    print(f"Missing titles: {stats['empty_title']}")
    print(f"Missing descriptions: {stats['empty_description']}")
    print(f"Missing authors: {stats['empty_author']}")
    print(f"Missing publish dates: {stats['empty_publish_date']}")
    print(f"Missing content: {stats['empty_content']}")
    print(f"Output file: {output_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    df = load_raw_data()
    raw_row_count = len(df)

    df = select_relevant_columns(df)
    df, unsuccessful_count = filter_successful_scrapes(df)
    df, full_dupe_count, url_dupe_count = remove_duplicates(df)
    df = clean_text_columns(df)
    df = handle_missing_values(df)
    df = clean_publish_date(df)
    df = clean_url(df)
    stats = validate_data(df)
    save_clean_data(df)

    print_final_report(raw_row_count, unsuccessful_count, full_dupe_count,
                        url_dupe_count, stats, CLEAN_CSV_PATH)

    print(f"\nFinal DataFrame columns: {list(df.columns)}")


if __name__ == "__main__":
    main()