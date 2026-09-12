import matplotlib.pyplot as plt
from pathlib import Path

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
predictions = model.predict(features_test)
print(f"Accuracy: {accuracy_score(target_test, predictions):.3f}")

# Visualization: compare class balance before and after the train/test split.
figure, axis = plt.subplots(figsize=(7, 4))
class_values = sorted(set(target))
axis.bar([value - 0.2 for value in class_values], [sum(target_train == value) for value in class_values], width=0.4, label="Train")
axis.bar([value + 0.2 for value in class_values], [sum(target_test == value) for value in class_values], width=0.4, label="Test")
axis.set(title="Class balance after splitting", xlabel="Class", ylabel="Sample count")
axis.legend()
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
