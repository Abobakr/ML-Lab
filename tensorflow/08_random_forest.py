from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


features, target = load_wine(return_X_y=True)
features_train, features_test, target_train, target_test = train_test_split(
	features, target, test_size=0.2, stratify=target, random_state=42
)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(features_train, target_train)

print(f"Random forest accuracy: {accuracy_score(target_test, model.predict(features_test)):.3f}")
print(f"Most important feature: {model.feature_importances_.argmax()}")
