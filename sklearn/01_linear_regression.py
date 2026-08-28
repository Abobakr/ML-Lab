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
