from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


features, target = load_iris(return_X_y=True)
features_train, features_test, target_train, target_test = train_test_split(
	features, target, test_size=0.2, stratify=target, random_state=42
)
model = GaussianNB()
model.fit(features_train, target_train)

print(f"Naive Bayes accuracy: {accuracy_score(target_test, model.predict(features_test)):.3f}")
