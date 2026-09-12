import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


features, target = load_iris(return_X_y=True)
model = Pipeline([("scale", StandardScaler()), ("classifier", SVC())])
search = GridSearchCV(
    model,
    {"classifier__C": [0.1, 1, 10], "classifier__kernel": ["linear", "rbf"]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy",
)
search.fit(features, target)
print(f"Best parameters: {search.best_params_}")
print(f"Best cross-validation accuracy: {search.best_score_:.3f}")

# Visualization: the grid makes the selected hyperparameters easy to compare.
results = search.cv_results_["mean_test_score"].reshape(3, 2)
figure, axis = plt.subplots(figsize=(7, 4))
image = axis.imshow(results, cmap="viridis", vmin=results.min(), vmax=results.max())
axis.set(xticks=range(2), xticklabels=["linear", "rbf"], yticks=range(3), yticklabels=["0.1", "1", "10"], title="SVC validation accuracy", xlabel="Kernel", ylabel="C")
figure.colorbar(image, ax=axis, label="Accuracy")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
