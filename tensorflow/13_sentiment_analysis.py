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
for text, sentiment in zip(examples, model.predict(examples)):
	print(f"{sentiment and 'positive' or 'negative'}: {text}")
