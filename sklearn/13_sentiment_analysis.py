import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


texts = [
	"This movie was excellent and inspiring",
	"A brilliant story with wonderful acting",
	"I loved every minute of this film",
	"The plot was engaging and enjoyable",
	"This was a fantastic experience",
	"The movie was boring and disappointing",
	"Terrible acting made this film painful",
	"I hated this slow and predictable story",
	"A poor movie with a pointless ending",
	"This film was awful and tedious",
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(random_state=42))
model.fit(texts, labels)

examples = ["an enjoyable and brilliant film", "a tedious and awful movie"]
predictions = model.predict(examples)
for text, sentiment in zip(examples, predictions):
	print(f"{sentiment and 'positive' or 'negative'}: {text}")

# Visualization: the strongest learned n-gram coefficients explain the predictions.
vectorizer = model.named_steps["tfidfvectorizer"]
classifier = model.named_steps["logisticregression"]
terms = vectorizer.get_feature_names_out()
strongest = classifier.coef_[0].argsort()
selected = list(strongest[:5]) + list(strongest[-5:])
figure, axis = plt.subplots(figsize=(9, 5))
axis.barh(terms[selected], classifier.coef_[0][selected], color=["steelblue" if classifier.coef_[0][index] < 0 else "indianred" for index in selected])
axis.set(title="Sentiment n-gram coefficients", xlabel="Coefficient")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
