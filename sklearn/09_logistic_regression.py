import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


features, target = load_breast_cancer(return_X_y=True)
features_train, features_test, target_train, target_test = train_test_split(
	features, target, test_size=0.2, stratify=target, random_state=42
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
model.fit(features_train, target_train)
predictions = model.predict(features_test)
probabilities = model.predict_proba(features_test)[:, 1]

print(classification_report(target_test, predictions, zero_division=0))

# Visualization: classification errors and threshold behavior are both visible.
false_positive_rate, true_positive_rate, _ = roc_curve(target_test, probabilities)
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(target_test, predictions, ax=axes[0], cmap="Blues")
axes[0].set(title="Logistic regression confusion matrix")
axes[1].plot(false_positive_rate, true_positive_rate)
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
