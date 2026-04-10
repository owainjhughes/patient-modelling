import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')


def _save(output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def _infer_features(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    return [c for c in num_cols if df[c].nunique() > 1]


def _subsample(X, n=20000):
    if len(X) <= n:
        return X
    rng = np.random.default_rng(42)
    return X[rng.choice(len(X), size=n, replace=False)]


def _find_optimal_k(X_scaled, max_k, output_dir, search_sample=5000):
    # Subsample so k-search stays fast; final model is still fit on full data
    if len(X_scaled) > search_sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_scaled), size=search_sample, replace=False)
        X_search = X_scaled[idx]
        print(f"  (k-search uses a {search_sample}-row sample for speed)")
    else:
        X_search = X_scaled

    k_range = range(2, min(max_k + 1, len(X_search)))
    inertias = []
    sil_scores = []

    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        labels = km.fit_predict(X_search)
        inertias.append(km.inertia_)
        # sample_size keeps silhouette O(n) instead of O(n²)
        sil_scores.append(silhouette_score(X_search, labels, sample_size=min(2000, len(X_search))))
        print(f"  k={k}  inertia={km.inertia_:.1f}  silhouette={sil_scores[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(k_range), inertias, marker='o')
    axes[0].set_title('Elbow Method — Inertia vs. k')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(True)

    axes[1].plot(list(k_range), sil_scores, marker='o', color='green')
    axes[1].set_title('Silhouette Score vs. k')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True)

    plt.tight_layout()
    _save(output_dir, 'kmeans_elbow_silhouette.png')

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"Best k by silhouette score: {best_k}  "
          f"(score={max(sil_scores):.3f})")
    return best_k


def _plot_clusters_2d(X_scaled, labels, feature_names, output_dir, plot_sample=5000):
    idx = np.random.default_rng(42).choice(len(X_scaled), size=min(plot_sample, len(X_scaled)), replace=False)
    X_plot, l_plot = X_scaled[idx], labels[idx]
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_plot)
    var = pca.explained_variance_ratio_

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1],
                          c=l_plot, cmap='tab10', alpha=0.5, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'KMeans Clusters (PCA 2-D projection)\n'
              f'PC1 explains {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    _save(output_dir, 'kmeans_clusters_2d.png')


def _plot_cluster_distribution(labels, output_dir):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8, 5))
    plt.bar([str(u) for u in unique], counts, color='steelblue', edgecolor='black')
    for i, (u, c) in enumerate(zip(unique, counts)):
        plt.text(i, c + counts.max() * 0.01, str(c), ha='center', va='bottom')
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.grid(True, axis='y')
    plt.tight_layout()
    _save(output_dir, 'kmeans_cluster_distribution.png')


def _plot_cluster_profiles(df_clustered, feature_cols, output_dir):
    profile = df_clustered.groupby('cluster')[feature_cols].mean()
    # Normalise to [0,1] per feature so all columns are visually comparable
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    plt.figure(figsize=(max(10, len(feature_cols)), max(4, len(profile))))
    sns.heatmap(profile_norm, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Cluster Profiles (normalised feature means)')
    plt.xlabel('Feature')
    plt.ylabel('Cluster')
    plt.tight_layout()
    _save(output_dir, 'kmeans_cluster_profiles.png')


def run_kmeans(input_csv, feature_cols=None, k=None, max_k=10, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"KMeans pipeline — loading: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)

    if feature_cols is None:
        feature_cols = _infer_features(df)
        print(f"No features specified — using numeric columns: {feature_cols}")

    if not feature_cols:
        raise ValueError("No usable numeric feature columns found. "
                         "Pass feature_cols explicitly.")

    X = df[feature_cols].dropna()
    print(f"Clustering on {len(feature_cols)} features, {len(X)} rows")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if k is None:
        print(f"\nSearching for optimal k in range [2, {max_k}]...")
        k = _find_optimal_k(X_scaled, max_k, output_dir)

    print(f"\nFitting MiniBatchKMeans with k={k}...")
    X_fit = _subsample(X_scaled)
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5)
    labels_fit = km.fit_predict(X_fit)
    # Predict labels for all rows using the fitted centres
    labels = km.predict(X_scaled)

    sil = silhouette_score(_subsample(X_scaled, 5000), km.predict(_subsample(X_scaled, 5000)))
    print(f"Silhouette score (k={k}, sampled): {sil:.3f}")

    df_clustered = X.copy()
    df_clustered['cluster'] = labels

    _plot_clusters_2d(X_scaled, labels, feature_cols, output_dir)
    _plot_cluster_distribution(labels, output_dir)
    _plot_cluster_profiles(df_clustered, feature_cols, output_dir)

    out_csv = os.path.join(output_dir, 'kmeans_labelled.csv')
    df_out = df.copy()
    df_out.loc[X.index, 'cluster'] = labels  # merge by index to preserve all original columns
    df_out.to_csv(out_csv, index=False)
    print(f"\nLabelled data saved to: {out_csv}")

    print("\nCluster summary:")
    print(df_clustered.groupby('cluster')[feature_cols].mean().round(3))

    print(f"{'='*60}\n")
    return df_clustered, km


if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else 'outputs/cleaned_data.csv'
    run_kmeans(csv, output_dir='outputs')
