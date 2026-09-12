import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


features, target = load_iris(return_X_y=True)
features_train, features_test, target_train, target_test = train_test_split(
	features, target, test_size=0.2, stratify=target, random_state=42
)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(features_train, target_train)
predictions = model.predict(features_test)

print(f"KNN accuracy: {accuracy_score(target_test, predictions):.3f}")

# Visualization: test samples colored by KNN's local-vote predictions.
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(features_test[:, 0], features_test[:, 1], c=predictions, cmap="viridis", edgecolors="black")
axes[0].set(title="KNN predicted classes", xlabel="Sepal length", ylabel="Sepal width")
ConfusionMatrixDisplay.from_predictions(target_test, predictions, ax=axes[1], cmap="Blues")
axes[1].set(title="KNN confusion matrix")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
