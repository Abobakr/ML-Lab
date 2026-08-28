from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


features, _ = load_digits(return_X_y=True)
scaled_features = StandardScaler().fit_transform(features)
model = PCA(n_components=0.95, random_state=42)
reduced_features = model.fit_transform(scaled_features)

print(f"Original dimensions: {features.shape[1]}")
print(f"Reduced dimensions: {reduced_features.shape[1]}")
print(f"Explained variance: {model.explained_variance_ratio_.sum():.3f}")
