from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


features, target = load_iris(return_X_y=True)
features_train, features_test, target_train, target_test = train_test_split(
	features, target, test_size=0.25, stratify=target, random_state=42
)
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(features_train, target_train)

print(f"Training samples: {len(features_train)}")
print(f"Test samples: {len(features_test)}")
print(f"Accuracy: {accuracy_score(target_test, model.predict(features_test)):.3f}")
