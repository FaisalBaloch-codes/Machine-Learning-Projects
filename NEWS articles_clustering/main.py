"""
main.py
--------
RUN THIS FILE: python3 main.py

Complete pipeline:
  STAGE 1 - Feature Engineering
    1. Load raw data (URL, Title, Category)
    2. Engineer 7 numeric structural features from Title text

  STAGE 2 - Clustering (unsupervised; Category is NEVER used to form clusters)
    3. Select clustering features, handle missing values, scale
    4. K-Means: test K=2..10, score each with Silhouette Score, pick best K
    5. Train final K-Means with the best K, save assignments + visualization
    6. Hierarchical (Agglomerative, Ward linkage): build dendrogram
    7. Select a distance threshold from the dendrogram (largest-gap method)
    8. Cut the tree at that threshold, save assignments
    9. Compare K-Means vs Hierarchical
   10. Optional post-clustering Category breakdown per cluster (interpretation only)
   11. Save everything to results/ and print a final summary
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd

from data_loader import load_dataset
from feature_engineering import add_engineered_features, ENGINEERED_FEATURE_COLUMNS
from preprocessing import select_clustering_features, scale_features
from evaluation import evaluate_k_range, select_best_k
from kmeans_clustering import run_kmeans
from hierarchical_clustering import compute_linkage, select_threshold, assign_clusters
from visualization import plot_silhouette_scores, plot_kmeans_clusters, plot_dendrogram
from analysis import cluster_sizes, cluster_category_breakdown

INPUT_FILE = "data/News_Classification_Labeling.xlsx"
RESULTS_DIR = "results"
K_RANGE = range(2, 11)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------- STAGE 1: FEATURE ENGINEERING ----------
    print("=" * 60)
    print("STAGE 1: FEATURE ENGINEERING")
    print("=" * 60)
    print("Loading raw dataset...")
    df = load_dataset(INPUT_FILE)

    print("Engineering structural features from Title text...")
    df = add_engineered_features(df)
    print(f"Engineered features: {ENGINEERED_FEATURE_COLUMNS}")

    # ---------- STAGE 2: CLUSTERING ----------
    print("\n" + "=" * 60)
    print("STAGE 2: CLUSTERING")
    print("=" * 60)

    print("Selecting clustering features (Category/URL/Title excluded)...")
    X = select_clustering_features(df)
    print("Scaling features (StandardScaler)...")
    X_scaled, scaler = scale_features(X)

    # --- K-Means: find best K ---
    print(f"\nTesting K-Means for K in {list(K_RANGE)} (Silhouette Score)...")
    scores_df = evaluate_k_range(X_scaled, K_RANGE)
    scores_df.to_csv(f"{RESULTS_DIR}/kmeans_scores.csv", index=False)
    best_k = select_best_k(scores_df)
    print(f"Best K selected: {best_k} "
          f"(highest Silhouette Score = {scores_df['Silhouette_Score'].max():.4f})")

    print(f"Training final K-Means with K={best_k}...")
    kmeans_model, kmeans_labels = run_kmeans(X_scaled, best_k)

    kmeans_out = df[["URL", "Title", "Category"]].copy()
    kmeans_out["Cluster"] = kmeans_labels
    for col in ENGINEERED_FEATURE_COLUMNS:
        kmeans_out[col] = df[col]
    kmeans_out.to_csv(f"{RESULTS_DIR}/kmeans_clustered_data.csv", index=False)

    kmeans_sizes = cluster_sizes(kmeans_labels)
    kmeans_sizes.to_csv(f"{RESULTS_DIR}/kmeans_cluster_sizes.csv", index=False)

    print("Generating K-Means visualizations...")
    plot_silhouette_scores(scores_df, f"{RESULTS_DIR}/silhouette_scores.png")
    plot_kmeans_clusters(X_scaled, kmeans_labels, f"{RESULTS_DIR}/kmeans_clusters.png", best_k)

    # --- Hierarchical clustering ---
    print("\nRunning Agglomerative Hierarchical Clustering (Ward linkage)...")
    Z = compute_linkage(X_scaled)
    threshold = select_threshold(Z)
    hier_labels = assign_clusters(Z, threshold)
    n_hier_clusters = len(set(hier_labels))
    print(f"Selected distance threshold: {threshold:.4f} -> {n_hier_clusters} clusters")

    print("Generating dendrogram...")
    plot_dendrogram(Z, f"{RESULTS_DIR}/hierarchical_dendrogram.png", threshold=threshold)

    with open(f"{RESULTS_DIR}/hierarchical_threshold.txt", "w") as f:
        f.write("Hierarchical Clustering - Threshold Selection\n")
        f.write("=" * 50 + "\n")
        f.write(f"Linkage method: Ward\n")
        f.write(f"Selection method: largest-gap heuristic on the last 15 merge distances\n")
        f.write(f"Selected distance threshold: {threshold:.4f}\n")
        f.write(f"Resulting number of clusters: {n_hier_clusters}\n")

    hier_out = df[["URL", "Title", "Category"]].copy()
    hier_out["Cluster"] = hier_labels
    for col in ENGINEERED_FEATURE_COLUMNS:
        hier_out[col] = df[col]
    hier_out.to_csv(f"{RESULTS_DIR}/hierarchical_clustered_data.csv", index=False)

    hier_sizes = cluster_sizes(hier_labels)
    hier_sizes.to_csv(f"{RESULTS_DIR}/hierarchical_cluster_sizes.csv", index=False)

    # --- Post-clustering category analysis (interpretation only, K-Means clusters) ---
    print("\nRunning optional post-clustering Category breakdown (K-Means clusters)...")
    kmeans_out_for_analysis = kmeans_out.rename(columns={"Cluster": "Cluster"})
    category_analysis = cluster_category_breakdown(kmeans_out_for_analysis, "Cluster")
    category_analysis.to_csv(f"{RESULTS_DIR}/cluster_category_analysis.csv", index=False)

    # --- Final summary ---
    summary_lines = []
    summary_lines.append("CLUSTERING PROJECT - FINAL SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Total rows used: {len(df)}")
    summary_lines.append(f"Engineered features used for clustering: {', '.join(ENGINEERED_FEATURE_COLUMNS)}")
    summary_lines.append("Category was excluded from clustering (unsupervised task).")
    summary_lines.append("")
    summary_lines.append("K-MEANS")
    summary_lines.append(f"  K values tested: {list(K_RANGE)}")
    summary_lines.append(f"  Best K: {best_k}")
    summary_lines.append(f"  Best Silhouette Score: {scores_df['Silhouette_Score'].max():.4f}")
    summary_lines.append(f"  Cluster sizes:\n{kmeans_sizes.to_string(index=False)}")
    summary_lines.append("")
    summary_lines.append("HIERARCHICAL (Agglomerative, Ward linkage)")
    summary_lines.append(f"  Selected distance threshold: {threshold:.4f}")
    summary_lines.append(f"  Resulting number of clusters: {n_hier_clusters}")
    summary_lines.append(f"  Cluster sizes:\n{hier_sizes.to_string(index=False)}")
    summary_lines.append("")
    summary_lines.append("NOTE: Clustering is unsupervised and does not recover the true")
    summary_lines.append("news Category labels. The category breakdown in")
    summary_lines.append("cluster_category_analysis.csv is for interpretation only.")

    summary_text = "\n".join(summary_lines)
    with open(f"{RESULTS_DIR}/clustering_summary.txt", "w") as f:
        f.write(summary_text)

    print("\n" + "=" * 60)
    print(summary_text)
    print("=" * 60)
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
