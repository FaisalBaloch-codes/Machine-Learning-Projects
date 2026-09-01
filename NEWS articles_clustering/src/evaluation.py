"""
evaluation.py
--------------
Computes the Silhouette Score for a range of candidate K values (for
K-Means) and selects the best K.

Silhouette Score, in plain terms:
  For each point, it compares (a) how close that point is to other points
  in its OWN cluster, vs (b) how close it is to points in the NEAREST other
  cluster. The score ranges from -1 to +1:
    - close to +1 : the point sits well inside its own cluster, far from
                     other clusters (good).
    - close to 0   : the point is right on the border between two clusters.
    - negative     : the point is probably in the wrong cluster.
  Averaged over all points, it gives one number per K -- higher is better.
  This is the ONE primary metric used here to pick K (as requested), rather
  than combining several internal metrics.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42


def evaluate_k_range(X_scaled, k_range=range(2, 11)) -> pd.DataFrame:
    """Fit K-Means for every K in k_range and record its Silhouette Score."""
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        rows.append({"K": k, "Silhouette_Score": round(score, 4)})
        print(f"[evaluation] K={k}: Silhouette Score = {score:.4f}")
    return pd.DataFrame(rows)


def select_best_k(scores_df: pd.DataFrame) -> int:
    best_row = scores_df.loc[scores_df["Silhouette_Score"].idxmax()]
    return int(best_row["K"])
