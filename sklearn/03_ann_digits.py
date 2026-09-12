from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


digits = load_digits()
features_train, features_test, target_train, target_test = train_test_split(
	digits.data, digits.target, test_size=0.2, stratify=digits.target, random_state=42
)
model = make_pipeline(
	StandardScaler(),
	MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True, random_state=42),
)
model.fit(features_train, target_train)

print(f"ANN digit accuracy: {accuracy_score(target_test, model.predict(features_test)):.3f}")
