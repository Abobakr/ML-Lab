from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


features, target = load_breast_cancer(return_X_y=True)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=42
)
model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42)
model.fit(features_train, target_train)
print(f"Gradient boosting accuracy: {accuracy_score(target_test, model.predict(features_test)):.3f}")
