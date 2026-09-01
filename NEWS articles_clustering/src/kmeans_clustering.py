"""
kmeans_clustering.py
----------------------
Trains the final K-Means model using the best K (selected in evaluation.py
via Silhouette Score) and returns the cluster label for every data point.
"""

from sklearn.cluster import KMeans

RANDOM_STATE = 42


def run_kmeans(X_scaled, k: int):
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels
