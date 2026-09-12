import matplotlib.pyplot as plt
from pathlib import Path

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

# Visualization: cluster colors and centroids show the learned assignments.
figure, axis = plt.subplots(figsize=(7, 5))
axis.scatter(features[:, 0], features[:, 1], c=labels, cmap="viridis", alpha=0.7)
axis.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", marker="X", s=180)
axis.set(title="K-Means clusters", xlabel="Feature 1", ylabel="Feature 2")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
