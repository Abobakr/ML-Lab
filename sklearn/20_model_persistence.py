from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


features, target = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=500, random_state=42).fit(features, target)
with TemporaryDirectory() as directory:
    path = Path(directory) / "iris_model.joblib"
    joblib.dump(model, path)
    restored_model = joblib.load(path)
    original_predictions = model.predict(features)
    restored_predictions = restored_model.predict(features)
    print(f"Saved model: {path.name}")
    print(f"Restored model accuracy: {restored_model.score(features, target):.3f}")

    # Visualization: matching predictions verify that persistence preserved behavior.
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(original_predictions, "o", label="Original")
    axis.plot(restored_predictions, "x", label="Restored")
    axis.set(title="Predictions before and after reload", xlabel="Sample", ylabel="Class")
    axis.legend()
    figure.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
