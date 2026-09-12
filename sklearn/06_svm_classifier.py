import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


features, target = load_breast_cancer(return_X_y=True)
features_train, features_test, target_train, target_test = train_test_split(
	features, target, test_size=0.2, stratify=target, random_state=42
)
model = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42))
model.fit(features_train, target_train)
predictions = model.predict(features_test)

print(f"SVM accuracy: {accuracy_score(target_test, predictions):.3f}")

# Visualization: use a confusion matrix because the true model space has 30 dimensions.
figure, axis = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(target_test, predictions, ax=axis, cmap="Blues")
axis.set(title="SVM confusion matrix")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
