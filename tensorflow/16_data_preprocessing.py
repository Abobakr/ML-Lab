from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
