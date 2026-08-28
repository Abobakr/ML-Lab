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
