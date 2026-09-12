import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


features, digits_target = load_digits(return_X_y=True)
scaled_features = StandardScaler().fit_transform(features)
model = PCA(n_components=0.95, random_state=42)
reduced_features = model.fit_transform(scaled_features)

print(f"Original dimensions: {features.shape[1]}")
print(f"Reduced dimensions: {reduced_features.shape[1]}")
print(f"Explained variance: {model.explained_variance_ratio_.sum():.3f}")

# Visualization: variance retention and the first two projected components.
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(model.explained_variance_ratio_.cumsum())
axes[0].axhline(0.95, color="black", linestyle="--")
axes[0].set(title="Cumulative explained variance", xlabel="Components", ylabel="Variance retained")
# Visualization: color points by their original digit labels for interpretation.
axes[1].scatter(reduced_features[:, 0], reduced_features[:, 1], c=digits_target, cmap="tab10", s=8)
axes[1].set(title="Digits in PCA space", xlabel="PC 1", ylabel="PC 2")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
