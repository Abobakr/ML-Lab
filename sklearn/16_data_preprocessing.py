import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# A small local dataset keeps this lesson reproducible and network-independent.
features = [[25, "junior"], [35, "senior"], [None, "junior"], [52, "senior"], [41, "senior"], [29, "junior"]]
target = [0, 1, 0, 1, 1, 0]
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([( "imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), [0]),
    ("category", OneHotEncoder(handle_unknown="ignore"), [1]),
])
model = Pipeline([("preprocess", preprocessor), ("classifier", LogisticRegression(random_state=42))])
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.33, random_state=42)
model.fit(features_train, target_train)
print(f"Preprocessing pipeline accuracy: {model.score(features_test, target_test):.3f}")

# Visualization: compare raw feature rows with the encoded pipeline output.
transformed_features = model.named_steps["preprocess"].transform(features)
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(len(features)), [row[0] if row[0] is not None else 0 for row in features])
axes[0].set(title="Raw numeric feature", xlabel="Row", ylabel="Value")
axes[1].imshow(transformed_features.toarray() if hasattr(transformed_features, "toarray") else transformed_features, aspect="auto", cmap="viridis")
axes[1].set(title="Transformed feature matrix", xlabel="Encoded feature", ylabel="Row")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
