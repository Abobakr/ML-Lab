import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


features, target = make_regression(
	n_samples=200, n_features=2, noise=12, random_state=42
)
features_train, features_test, target_train, target_test = train_test_split(
	features, target, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(features_train, target_train)
predictions = model.predict(features_test)

print(f"R2: {r2_score(target_test, predictions):.3f}")
print(f"RMSE: {mean_squared_error(target_test, predictions) ** 0.5:.3f}")

# Visualization: compare held-out predictions with the observed targets.
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(target_test, predictions, alpha=0.8)
axes[0].plot([target_test.min(), target_test.max()], [target_test.min(), target_test.max()], "k--")
axes[0].set(title="Actual vs predicted", xlabel="Actual target", ylabel="Predicted target")
axes[1].scatter(predictions, target_test - predictions, alpha=0.8)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set(title="Residuals", xlabel="Predicted target", ylabel="Residual")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
