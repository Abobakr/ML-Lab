from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


features, _ = make_blobs(
	n_samples=300, centers=3, cluster_std=0.75, random_state=42
)
model = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = model.fit_predict(features)

print(f"Cluster centers:\n{model.cluster_centers_}")
print(f"Silhouette score: {silhouette_score(features, labels):.3f}")
