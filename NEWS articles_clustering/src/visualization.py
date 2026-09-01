"""
visualization.py
-------------------
All plots for the project. Uses matplotlib (+ scipy's dendrogram helper).

IMPORTANT NOTE ON PCA:
  The engineered feature space has 7 dimensions, which can't be plotted
  directly. PCA is used HERE ONLY to project the 7D data down to 2D for
  visualization. Clustering itself (K-Means and hierarchical) is performed
  on the full 7D scaled feature space, NOT on the PCA components -- PCA is
  a visualization aid only, not part of the clustering pipeline.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram


def plot_silhouette_scores(scores_df, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scores_df["K"], scores_df["Silhouette_Score"], marker="o", color="#2E5090")
    best_k = scores_df.loc[scores_df["Silhouette_Score"].idxmax(), "K"]
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.6, label=f"Best K = {int(best_k)}")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs K (K-Means)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_kmeans_clusters(X_scaled, labels, save_path, k):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=12, alpha=0.7)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title(
        f"K-Means Clusters (K={k}), visualized via PCA\n"
        f"(clustering was done on the full 7D feature space; PCA 2D is for\n"
        f"viewing only and captures {explained:.1%} of the variance)",
        fontsize=11
    )
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_dendrogram(Z, save_path, threshold=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, no_labels=True, color_threshold=threshold, ax=ax)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--",
                    label=f"Selected threshold = {threshold:.2f}")
        ax.legend()
    ax.set_xlabel("Data points (individual titles)")
    ax.set_ylabel("Distance (Ward linkage)")
    ax.set_title("Hierarchical Clustering Dendrogram")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
