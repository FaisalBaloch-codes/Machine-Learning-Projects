"""
analysis.py
-------------
Post-clustering, interpretation-only analysis. Category was NEVER used to
form the clusters -- this module only looks at how the (unsupervised)
clusters happen to line up with the (known, unused-during-clustering)
Category labels, purely to help a human interpret what each cluster
might represent.
"""

import pandas as pd


def cluster_sizes(labels) -> pd.DataFrame:
    counts = pd.Series(labels).value_counts().sort_index()
    return counts.rename_axis("Cluster").reset_index(name="Size")


def cluster_category_breakdown(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """For each cluster, what % of its rows belong to each known Category.
    Rows with a missing Category are excluded from this table only."""
    valid = df.dropna(subset=["Category"])
    crosstab = pd.crosstab(valid[cluster_col], valid["Category"], normalize="index") * 100
    crosstab = crosstab.round(1)
    crosstab.index.name = "Cluster"
    return crosstab.reset_index()
