"""
preprocessing.py
------------------
STAGE 2 prep: selects the columns that are valid clustering INPUT features,
explicitly excludes identifiers/labels, handles missing values, and scales.

Excluded from clustering (and why):
  - URL       : identifier, not a numeric feature
  - Title     : raw text, not a numeric feature (already converted into
                the engineered numeric features above)
  - Category  : the known label. Clustering is unsupervised -- using the
                label to build clusters would defeat the purpose and leak
                target information into the "discovery" process. Category
                is preserved separately and only reattached AFTER clustering,
                for interpretation (see analysis.py).

Why scaling is necessary:
  The 7 engineered features have very different natural ranges, e.g.
  title_len is roughly 16-275, num_digits_count is roughly 0-23, while
  sentiment_score is bounded -1 to 1 and has_location is 0/1. K-Means and
  Agglomerative (Euclidean/Ward) clustering both compute distances between
  points -- without scaling, title_len alone would dominate every distance
  calculation and the other 6 features would barely influence the clusters.
  StandardScaler puts every feature on a comparable scale (mean 0, std 1)
  so each feature contributes proportionally.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_engineering import ENGINEERED_FEATURE_COLUMNS

EXCLUDED_COLUMNS = ["URL", "Title", "Category"]


def select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the numeric engineered feature columns, with missing
    values (if any) filled with 0."""
    X = df[ENGINEERED_FEATURE_COLUMNS].copy()
    if X.isnull().any().any():
        n = X.isnull().sum().sum()
        print(f"[preprocessing] Filling {n} missing engineered-feature value(s) with 0.")
        X = X.fillna(0)
    return X


def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
