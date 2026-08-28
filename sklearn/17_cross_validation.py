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
