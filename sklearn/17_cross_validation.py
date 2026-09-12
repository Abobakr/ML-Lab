import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


features, target = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, features, target, cv=folds, scoring="accuracy")
print(f"Fold accuracies: {scores.round(3)}")
print(f"Mean accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

# Visualization: fold scores show how much validation performance varies.
figure, axis = plt.subplots(figsize=(7, 4))
axis.plot(range(1, len(scores) + 1), scores, marker="o")
axis.axhline(scores.mean(), color="black", linestyle="--", label="Mean")
axis.set(title="Cross-validation fold accuracy", xlabel="Fold", ylabel="Accuracy")
axis.legend()
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
