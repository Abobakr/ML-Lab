import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split


features, target = load_wine(return_X_y=True)
features_train, features_test, target_train, target_test = train_test_split(
	features, target, test_size=0.2, stratify=target, random_state=42
)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(features_train, target_train)
predictions = model.predict(features_test)

print(f"Random forest accuracy: {accuracy_score(target_test, predictions):.3f}")
print(f"Most important feature: {model.feature_importances_.argmax()}")

# Visualization: importance bars show which input features the forest used most.
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(features.shape[1]), model.feature_importances_)
axes[0].set(title="Random forest feature importance", xlabel="Feature index", ylabel="Importance")
ConfusionMatrixDisplay.from_predictions(target_test, predictions, ax=axes[1], cmap="Blues")
axes[1].set(title="Random forest confusion matrix")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
