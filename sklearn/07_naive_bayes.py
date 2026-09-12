import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


features, target = load_iris(return_X_y=True)
features_train, features_test, target_train, target_test = train_test_split(
	features, target, test_size=0.2, stratify=target, random_state=42
)
model = GaussianNB()
model.fit(features_train, target_train)
predictions = model.predict(features_test)

print(f"Naive Bayes accuracy: {accuracy_score(target_test, predictions):.3f}")

# Visualization: feature distributions make the Gaussian class assumption visible.
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
for class_value in sorted(set(target)):
	axes[0].hist(features[target == class_value, 0], alpha=0.5, label=f"Class {class_value}")
axes[0].set(title="Naive Bayes feature distributions", xlabel="Sepal length", ylabel="Count")
axes[0].legend()
ConfusionMatrixDisplay.from_predictions(target_test, predictions, ax=axes[1], cmap="Blues")
axes[1].set(title="Naive Bayes confusion matrix")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
