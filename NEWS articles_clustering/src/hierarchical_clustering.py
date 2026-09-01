"""
hierarchical_clustering.py
-----------------------------
Agglomerative (bottom-up) hierarchical clustering:
  - Every point starts as its own cluster.
  - At each step, the two closest clusters are merged.
  - This repeats until everything is merged into one cluster, forming a
    tree (dendrogram).
  Ward linkage is used: it merges the pair of clusters that leads to the
  smallest increase in total within-cluster variance -- it tends to produce
  compact, evenly-sized clusters, similar in spirit to K-Means.

Threshold selection:
  Cutting the dendrogram at a given DISTANCE produces a specific number of
  clusters. Rather than hardcoding an arbitrary distance, this uses the
  "largest gap" heuristic: look at the last N merge distances (the highest,
  most significant merges near the top of the tree) and find the biggest
  jump between two consecutive merge distances. Everything below that gap
  is considered "genuinely close"; everything above it is "forcing unlike
  clusters together". The threshold is set at the midpoint of that gap.
  This is a standard, explainable way to pick a cut point directly from the
  data instead of guessing a number.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

LINKAGE_METHOD = "ward"


def compute_linkage(X_scaled):
    return linkage(X_scaled, method=LINKAGE_METHOD)


def select_threshold(Z, top_n_merges: int = 15) -> float:
    """Largest-gap heuristic: find the biggest jump among the last
    top_n_merges merge distances and set the threshold at the gap's
    midpoint."""
    merge_distances = Z[:, 2]
    last_merges = merge_distances[-top_n_merges:]
    gaps = np.diff(last_merges)
    gap_idx = np.argmax(gaps)
    threshold = (last_merges[gap_idx] + last_merges[gap_idx + 1]) / 2
    return float(threshold)


def assign_clusters(Z, threshold: float):
    labels = fcluster(Z, t=threshold, criterion="distance")
    return labels
