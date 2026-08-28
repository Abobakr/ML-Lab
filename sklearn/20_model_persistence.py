from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


features, target = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=500, random_state=42).fit(features, target)
with TemporaryDirectory() as directory:
    path = Path(directory) / "iris_model.joblib"
    joblib.dump(model, path)
    restored_model = joblib.load(path)
    print(f"Saved model: {path.name}")
    print(f"Restored model accuracy: {restored_model.score(features, target):.3f}")
