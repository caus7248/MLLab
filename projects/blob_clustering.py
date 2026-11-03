"""Cluster generated points with K-means."""

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


def run():
    """Fit K-means and return the silhouette score of its clusters."""
    features, _ = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=0.6,
        random_state=42,
    )
    model = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = model.fit_predict(features)
    return silhouette_score(features, labels)


if __name__ == "__main__":
    print(f"Blob clustering silhouette score: {run():.3f}")
