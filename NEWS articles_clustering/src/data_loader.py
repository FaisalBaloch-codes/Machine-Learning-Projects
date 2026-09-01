"""
data_loader.py
---------------
Loads the raw news dataset (URL, Title, Category).
"""

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Load the Excel dataset and drop any row with a missing Title
    (Title is required to engineer features). Category may be missing for
    a few rows -- that's fine, since Category is never used to build
    clusters; it's only used later for optional post-clustering analysis
    (and rows with missing Category are simply skipped in that analysis).
    """
    df = pd.read_excel(path)
    before = len(df)
    df = df.dropna(subset=["Title"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[data_loader] Dropped {dropped} row(s) with missing Title.")
    if df["Category"].isnull().any():
        n = df["Category"].isnull().sum()
        print(f"[data_loader] Note: {n} row(s) have missing Category "
              f"(kept for clustering, excluded from category analysis).")
    return df
