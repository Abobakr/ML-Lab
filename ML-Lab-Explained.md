# ML-Lab Explained — A Beginner's Guide to Every Line of Code

**Who this is for:** you know Python (variables, functions, loops, `import`) but nothing about machine learning (ML). By the end of this document you will understand *what every single line of every lesson does, and why it's there.*

---

## Part 0 — The Big Picture (read this before anything else)

### What is Machine Learning, really?

Traditional programming: **you** write the rules.
```python
if temperature > 30:
    print("It's hot")
```

Machine Learning: **the computer learns the rules** from examples (data), instead of you writing them by hand.

You give it:
- **Features** — the input information (e.g. house size, number of rooms)
- **Target** (a.k.a. **label**) — the answer you want it to predict (e.g. house price)

The computer looks at hundreds/thousands of (features, target) pairs and finds a mathematical pattern connecting them. Once it has that pattern (called a **model**), you can give it *new* features it's never seen and it will guess the target.

### Two libraries, two philosophies

This repo has two tracks that solve the *same* problems two different ways:

| | `sklearn/` (scikit-learn) | `tensorflow/` (TensorFlow/Keras) |
|---|---|---|
| Best for | "Classical" ML: statistics-based algorithms | Deep Learning: neural networks |
| Style | `model.fit(X, y)` then `model.predict(X)` — very short | You *design* a network layer by layer |
| Speed on small data | Fast | Often overkill/slower |
| Speed on huge, messy data (images, text, audio) | Struggles | Excels |

Both tracks use the exact same **workflow skeleton** in almost every file:
1. Get data (`features`, `target`)
2. Split data into **training set** and **test set**
3. Create a model
4. `model.fit(...)` — teach it using training data
5. `model.predict(...)` — ask it to guess on test data
6. Measure how good the guesses were (**metrics**)

Keep this skeleton in your head. Every lesson below is a variation of it.

### Why do we split data into train/test?

If you teach the model using the *exact same* data you test it on, it's like giving a student the exam questions before the exam — of course they'll "pass." Splitting the data means we test the model on examples it has **never seen**, which tells us if it actually learned a general pattern or just memorized answers.

### What is `random_state=42`?

Many operations here involve randomness (shuffling data, initializing weights). `random_state=42` (the number 42 is a programmer in-joke, it could be any number) **fixes** that randomness so that every time you run the code, you get the *exact same* result. This is called a **seed**. Without it, your R² score might be 0.89 one run and 0.85 the next — annoying when learning or debugging.

### What is `.astype("float32")` and why divide by 255.0?

Images are stored as pixel brightness values from 0 (black) to 255 (white), as whole numbers (integers). Neural networks train much better on small decimal numbers (this is called **normalization**). So:
```python
features_train.astype("float32") / 255.0
```
- `.astype("float32")` — converts integers to decimal (floating-point) numbers
- `/ 255.0` — squashes the range from [0, 255] down to [0.0, 1.0]

---

## Part 1 — The `sklearn/` Track (Classical Machine Learning)

### Lesson 01 — Linear Regression (`01_linear_regression.py`)

**Goal:** predict a continuous number (like a price or temperature) from input features, assuming a *straight-line* relationship.

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
```
- `make_regression` — a **fake data generator** built into sklearn. Real projects use real datasets (CSV files, databases); this lesson invents numbers so you don't need internet access or a dataset file.
- `LinearRegression` — the model class: fits a straight line (in higher dimensions, a "hyperplane") through the data.
- `mean_squared_error`, `r2_score` — two ways of scoring "how good was the prediction."
- `train_test_split` — splits data into training/testing portions.

```python
features, target = make_regression(
	n_samples=200, n_features=2, noise=12, random_state=42
)
```
- `n_samples=200` — generate 200 fake data points (rows).
- `n_features=2` — each data point has 2 input numbers (imagine: house size, house age).
- `noise=12` — real-world data is never perfectly clean; this adds random "wobble" so the relationship isn't a perfect mathematical line (more realistic).
- `random_state=42` — reproducibility, as explained in Part 0.
- Returns two arrays: `features` (shape 200×2 — 200 rows, 2 columns) and `target` (shape 200 — 200 numbers, the "correct answers").

```python
features_train, features_test, target_train, target_test = train_test_split(
	features, target, test_size=0.2, random_state=42
)
```
- `test_size=0.2` — reserve **20%** of the data (40 rows) for testing; the other 80% (160 rows) is for training.
- This function conveniently returns **4** arrays at once, splitting both `features` and `target` the same way so rows stay matched up correctly.

```python
model = LinearRegression()
model.fit(features_train, target_train)
```
- `LinearRegression()` — creates an "empty," untrained model object.
- `.fit(X, y)` — the **learning step**. Internally, sklearn uses linear algebra to find the best-fitting line/plane through `(features_train, target_train)`. After this line runs, `model` now "knows" the pattern.

```python
predictions = model.predict(features_test)
```
- `.predict(X)` — feeds the **test** features (which the model has never seen) through the learned line, and outputs a guessed number for each row.

```python
print(f"R2: {r2_score(target_test, predictions):.3f}")
print(f"RMSE: {mean_squared_error(target_test, predictions) ** 0.5:.3f}")
```
- **R² (R-squared)**: ranges (roughly) 0 to 1. It answers "what fraction of the variation in the target does my model explain?" 1.0 = perfect predictions, 0.0 = no better than always guessing the average.
- **RMSE (Root Mean Squared Error)**: the average size of the mistake, in the same units as your target. `mean_squared_error` computes the *squared* error (squaring punishes big mistakes more; also makes negative errors positive), then `** 0.5` takes the square root to bring it back to normal units.
- `:.3f` — a Python f-string formatting flag: prints the number rounded to 3 decimal places.

---

### Lesson 02 — K-Means Clustering (`02_kmeans_clustering.py`)

**New concept: unsupervised learning.** Every lesson so far had a `target` — the "correct answer" to learn from (this is called **supervised learning**). Clustering has **no target at all**. You just give the model data points and ask it to find natural groups on its own.

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
```
- `KMeans` — the clustering algorithm.
- `make_blobs` — fake-data generator that creates points clumped into visually separate "blobs" (like clusters of stars).
- `silhouette_score` — a metric that scores how well-separated the found clusters are.

```python
features, _ = make_blobs(
	n_samples=300, centers=3, cluster_std=0.75, random_state=42
)
```
- `centers=3` — generate 3 distinct blobs of points.
- `cluster_std=0.75` — how spread-out (loose vs. tight) each blob is. Smaller = tighter clusters.
- `_` — Python convention meaning "I'm receiving a value here but I don't need it." `make_blobs` also returns the *true* cluster labels, but we throw them away — because in real clustering problems, you never know the true groups in advance (that's the whole point of clustering).

```python
model = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = model.fit_predict(features)
```
- `n_clusters=3` — **you** must tell K-Means how many groups to look for. It doesn't guess this number itself.
- `n_init=10` — K-Means starts from random guesses and can get "unlucky" and land on a mediocre grouping. This runs the whole algorithm 10 times with different random starting points and keeps the best result.
- `.fit_predict(features)` — combines training (`fit`) and immediately getting the cluster assignment for each point (`predict`) in one call. `labels` will be an array like `[0, 2, 1, 0, 0, 2, ...]` — one cluster number per data point.

```python
print(f"Cluster centers:\n{model.cluster_centers_}")
print(f"Silhouette score: {silhouette_score(features, labels):.3f}")
```
- `model.cluster_centers_` — after training, K-Means stores the (x, y) coordinates of each cluster's "center of mass." The trailing underscore (`_`) is a scikit-learn convention meaning "this attribute only exists after `.fit()` has been called."
- **Silhouette score**: ranges from -1 to 1. Close to 1 = points are tightly grouped with their own cluster and far from other clusters (good). Close to 0 = clusters overlap. Negative = points are probably in the wrong cluster.

---

### Lesson 03 — ANN on Digits (`03_ann_digits.py`, sklearn version)

**New concept: Artificial Neural Network (ANN)**, and **classification** (predicting a category, not a number).

An ANN is loosely inspired by brain neurons. It's built from **layers** of simple math units. Each unit takes numbers in, multiplies them by learned "importance" numbers (**weights**), adds them up, and passes the result through a small nonlinear function. Stacking many of these lets the network learn very complex patterns — like recognizing handwritten digits.

```python
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```
- `load_digits` — a small **built-in** dataset: 1,797 images of handwritten digits (0–9), each only 8×8 pixels (tiny, so it trains fast).
- `MLPClassifier` — "Multi-Layer Perceptron," scikit-learn's basic neural network class for classification.
- `StandardScaler` — rescales each input number to have mean 0 and spread (standard deviation) 1. Neural networks train much better/faster when inputs are on a similar, small scale, instead of e.g. 0–255.
- `make_pipeline` — chains multiple processing steps together so they always run in the same order, on both training and test data, without you manually repeating code.

```python
digits = load_digits()
features_train, features_test, target_train, target_test = train_test_split(
	digits.data, digits.target, test_size=0.2, stratify=digits.target, random_state=42
)
```
- `digits.data` — the pixel values (flattened 8×8 image = 64 numbers per row).
- `digits.target` — the correct digit label (0–9) for each image.
- `stratify=digits.target` — **important new flag.** Normally `train_test_split` splits randomly, which could by bad luck put almost all "7"s in the training set and almost none in the test set. `stratify` forces the split to preserve the *same proportion* of each digit (0–9) in both train and test sets — a fairer, more reliable split for classification problems.

```python
model = make_pipeline(
	StandardScaler(),
	MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True, random_state=42),
)
```
- This builds a two-step pipeline: first scale the data, then feed it to the neural network — automatically, every time `.fit()` or `.predict()` is called.
- `hidden_layer_sizes=(128,)` — the network has **one hidden layer** with **128 neurons**. (A tuple like `(128, 64)` would mean two hidden layers.)
- `max_iter=300` — train for at most 300 passes over the data (each full pass is called an **epoch**).
- `early_stopping=True` — automatically **stop training early** if the model stops improving, to avoid wasting time or overfitting (see Part 0 concepts below).

```python
model.fit(features_train, target_train)
print(f"ANN digit accuracy: {accuracy_score(target_test, model.predict(features_test)):.3f}")
```
- **Accuracy** — the simplest classification metric: `(number of correct predictions) / (total predictions)`. An accuracy of 0.95 means 95% of test images were correctly identified.

---

### Lesson 04 — Train/Test Split, in depth (`04_train_test_split.py`)

This lesson exists purely to make the train/test concept explicit and visible (earlier lessons used it too, but this one prints the sizes).

```python
from sklearn.datasets import load_iris
```
- `load_iris` — the most famous toy dataset in all of ML: 150 flowers, 4 measurements each (petal/sepal length & width), 3 species to classify.

```python
features_train, features_test, target_train, target_test = train_test_split(
	features, target, test_size=0.25, stratify=target, random_state=42
)
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(features_train, target_train)

print(f"Training samples: {len(features_train)}")
print(f"Test samples: {len(features_test)}")
print(f"Accuracy: {accuracy_score(target_test, model.predict(features_test)):.3f}")
```
- `test_size=0.25` → with 150 flowers, this gives ~112 training rows and ~38 test rows.
- **Logistic Regression** (see lesson 09 for full detail) is used here as a simple, fast classifier just to demonstrate the split — despite the name "regression," it's actually used for classification.
- `len(features_train)` — literally just counts how many rows ended up in each split, so you can *see* the 75/25 split in the printed output.

---

### Lesson 05 — K-Nearest Neighbors / KNN (`05_knn_classifier.py`)

**New concept:** KNN is possibly the simplest ML algorithm to understand intuitively: *"to classify a new point, look at its `k` closest neighbors in the training data, and take a majority vote."*

```python
from sklearn.neighbors import KNeighborsClassifier
```

```python
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
```
- `n_neighbors=5` — for every new flower to classify, KNN finds the 5 closest flowers (by measurement distance) in the training set and predicts whatever species is most common among those 5.
- **Why `StandardScaler` matters here more than almost anywhere else:** KNN's "closeness" is calculated using raw numeric distance. If one feature is measured in centimeters (0–10) and another in millimeters (0–100), the millimeter feature would dominate the distance calculation purely due to scale, not actual importance. Scaling puts every feature on equal footing.

The rest (`fit`, `predict`, `accuracy_score`) works exactly like Lesson 04 — this is the repeating skeleton from Part 0 in action.

---

### Lesson 06 — Support Vector Machine / SVM (`06_svm_classifier.py`)

**New concept:** SVM tries to draw the **best possible dividing boundary** between classes — specifically, the boundary that maximizes the *margin* (gap) between the boundary and the nearest data points of each class.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
```
- `load_breast_cancer` — a real medical dataset: 30 measurements from tumor scans, label = malignant or benign.
- `SVC` — "Support Vector Classifier."

```python
model = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42))
```
- `kernel="rbf"` — SVM can draw straight boundaries (`kernel="linear"`) or curved, flexible boundaries. `"rbf"` (Radial Basis Function) is the most common choice for **non-linear**, curved boundaries — it can separate classes that a straight line cannot.

---

### Lesson 07 — Naive Bayes (`07_naive_bayes.py`)

**New concept:** a classifier based purely on **probability** (Bayes' Theorem). It's called "naive" because it makes a simplifying (and technically often wrong, but usually good-enough) assumption: that all input features are *independent* of each other.

```python
from sklearn.naive_bayes import GaussianNB
```
- `GaussianNB` — assumes each feature, within each class, follows a **Gaussian** (bell-curve/normal) distribution. It calculates, for a new flower, "given these measurements, which species is *most probable*?"

```python
model = GaussianNB()
```
- Notice: **no `StandardScaler` here.** Naive Bayes computes probability distributions per feature directly; scaling doesn't change probability rankings the way it changes raw distance calculations in KNN/SVM, so it's typically skipped.

---

### Lesson 08 — Random Forest (`08_random_forest.py`)

**New concept: ensembles** — combining many simple models into one stronger model. A Random Forest is a large collection of **Decision Trees** (a decision tree = a series of yes/no questions, like "is petal length > 2.5cm?").

```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
```
- `load_wine` — 178 wine samples, 13 chemical measurements, 3 wine classes.

```python
model = RandomForestClassifier(n_estimators=200, random_state=42)
```
- `n_estimators=200` — build **200 different decision trees**, each trained on a slightly different random subset of the data/features. For a prediction, all 200 trees vote, and the majority wins. This "wisdom of crowds" approach is far more accurate and less prone to memorizing noise than a single tree.

```python
print(f"Most important feature: {model.feature_importances_.argmax()}")
```
- `.feature_importances_` — after training, the forest can tell you how much each of the 13 chemical measurements contributed to its decisions (a big benefit of tree-based models — many ML models are "black boxes" that can't explain themselves this easily).
- `.argmax()` — a NumPy method meaning "give me the **index** of the largest value in this array" (not the value itself, the *position*). So this prints "which feature number (0–12) mattered most."

---

### Lesson 09 — Logistic Regression (`09_logistic_regression.py`)

**Clearing up the name confusion:** despite "regression" in the name, Logistic Regression is a **classification** algorithm. It works by fitting a straight-line-like formula, then squashing the result through a special S-shaped curve (the **sigmoid function**) that outputs a probability between 0 and 1 — e.g., "92% chance this tumor is malignant."

```python
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
```
- `max_iter=1000` — Logistic Regression trains iteratively (gradually improving its formula over many small steps, called **gradient descent** — more on this in Part 2). `max_iter` caps how many improvement steps it's allowed before giving up, even if it hasn't perfectly converged yet.

```python
print(classification_report(target_test, model.predict(features_test), zero_division=0))
```
- `classification_report` — instead of one single accuracy number, this prints a detailed table with, **for each class**:
  - **Precision**: "of everything I labeled malignant, what % was actually malignant?" (avoiding false alarms)
  - **Recall**: "of everything that was actually malignant, what % did I catch?" (avoiding missed cases)
  - **F1-score**: a single number balancing precision and recall together.
- `zero_division=0` — if a class has zero predicted samples, precision/recall math would involve dividing by zero (undefined). This flag tells sklearn to just print `0` instead of crashing or throwing a warning.

---

### Lessons 10, 12, 14 in the sklearn folder (RNN, CNN, Transformers)

These three files are **placeholders** — one-line comments that simply say the real implementation lives in the `tensorflow/` folder instead. Why? Because Recurrent Neural Networks, Convolutional Neural Networks, and Transformers are **deep learning** architectures that scikit-learn's toolkit isn't designed to build — you genuinely need a framework like TensorFlow (or PyTorch) for them. Skip ahead to Part 2, Lessons 10/12/14, for the real explanations.

---

### Lesson 11 — Q-Learning (`11_q_learning.py`, sklearn-track / tabular version)

**New concept: Reinforcement Learning (RL).** This is a completely different paradigm from everything above:
- There's no fixed dataset of (features, target) pairs.
- Instead, an **agent** takes **actions** inside an **environment**, receives a **reward** (or penalty) after each action, and gradually learns which actions lead to the best long-term reward — through trial and error, like training a dog with treats.

This file technically imports `gymnasium`, not `sklearn` — it's grouped here because it represents the "classical"/tabular approach to RL, in parallel with the neural-network version in the tensorflow folder.

```python
import numpy as np
import gymnasium as gym
```
- **Gymnasium** — a standard library providing pre-built simulated environments (games, robots, grids) for practicing RL.

```python
environment = gym.make("CliffWalking-v1")
```
- **CliffWalking** is a grid-world puzzle: an agent stands on a grid with a cliff along one edge. It must walk from a start square to a goal square without falling off the cliff (big penalty) — while also minimizing the number of steps taken (small penalty per step, to discourage wandering).

```python
q_table = np.zeros((environment.observation_space.n, environment.action_space.n))
```
- The **Q-table** is the "brain" of this simple agent: a big grid (2D array) where each row = one possible grid position (**state**), and each column = one possible move (**action** — up/down/left/right). Each cell holds a number: "how good is it, on average, to take this action from this state?" This number is called a **Q-value**.
- `environment.observation_space.n` — the total number of distinct states (grid squares) in this environment.
- `environment.action_space.n` — the total number of possible actions (4: up/down/left/right).
- `np.zeros(...)` — start with *no knowledge at all*: every Q-value begins at 0. The agent learns everything through experience.

```python
learning_rate = 0.1
discount_factor = 0.95
```
- **Learning rate (0.1)** — how much to update our belief after each new piece of experience. Small = cautious/slow learning (more stable). Large = fast but jumpy learning.
- **Discount factor (0.95)** — how much the agent cares about *future* rewards vs. *immediate* ones. Close to 1 = "think long-term, plan ahead." Close to 0 = "only care about the very next reward."

```python
for episode in range(5000):
```
- An **episode** = one full attempt at the task, from start to either reaching the goal or falling off the cliff. The agent plays through 5,000 full attempts, learning a little more each time.

```python
	observation, _ = environment.reset(seed=episode)
```
- Resets the agent back to the starting square at the beginning of each episode. `seed=episode` gives each episode's random elements a different but reproducible seed.

```python
	finished = False
	while not finished:
		exploration_rate = max(0.05, 1 - episode / 4000)
```
- **The exploration vs. exploitation tradeoff** — one of the most important ideas in all of RL:
  - **Exploitation**: use what you've already learned, pick the action with the best-known Q-value.
  - **Exploration**: try something random, in case there's a better strategy you haven't discovered yet.
- This line calculates a probability that **decreases over time**: at `episode=0`, `exploration_rate = 1 - 0/4000 = 1.0` (100% random — makes sense, the agent knows nothing yet). By `episode=4000`, it hits `1 - 4000/4000 = 0`, but `max(0.05, ...)` puts a **floor** of 5% — so the agent always keeps a small chance of trying something new, forever.

```python
		if random_generator.random() < exploration_rate:
			action = environment.action_space.sample()
		else:
			action = int(np.argmax(q_table[observation]))
```
- `random_generator.random()` — draws a random decimal between 0 and 1.
- If it's less than our current `exploration_rate`, **explore**: `environment.action_space.sample()` picks a completely random action.
- Otherwise, **exploit**: `q_table[observation]` looks up the row of Q-values for the current state; `np.argmax(...)` picks the *index* (i.e., the action) with the highest Q-value — "the best move we currently believe in."

```python
		next_observation, reward, terminated, truncated, _ = environment.step(action)
```
- `.step(action)` — actually performs the action in the environment and returns:
  - `next_observation` — the new grid square the agent landed on.
  - `reward` — points earned/lost (e.g., -1 per step, -100 for falling off the cliff, 0 or positive at the goal).
  - `terminated` — `True` if the episode ended "naturally" (reached goal or fell off cliff).
  - `truncated` — `True` if the episode was cut off for an external reason (e.g., a time limit), unrelated to success/failure.

```python
		best_future_value = 0 if terminated else np.max(q_table[next_observation])
```
- If the episode just ended, there is no "future" left to consider, so `best_future_value = 0`.
- Otherwise, look up the best Q-value achievable from the *new* square — "assuming I play optimally from here on, how good does the future look?"

```python
		q_table[observation, action] += learning_rate * (
			reward + discount_factor * best_future_value - q_table[observation, action]
		)
```
- **This is the core Q-learning update formula (the Bellman equation).** Let's break it apart:
  - `reward + discount_factor * best_future_value` — this is the "new evidence": immediate reward, plus a discounted estimate of future reward.
  - `... - q_table[observation, action]` — subtract what we *currently* believe. This difference is called the **TD-error** (Temporal Difference error): "how wrong was my previous estimate?"
  - `learning_rate * (...)` — only nudge our belief by a fraction (10%) of that error, rather than fully overwriting it, so learning is smooth and stable rather than erratic.
  - `q_table[...] +=` — apply that nudge to update the table entry.
- This single line, run millions of times across all episodes, is *the entire learning process*. There's no neural network, no gradient descent — just repeatedly refining a big lookup table based on experience.

```python
		observation = next_observation
		finished = terminated or truncated
```
- Move the "current position" pointer forward, and check if the episode is over.

```python
observation, _ = environment.reset(seed=123)
total_reward = 0
for _ in range(200):
	action = int(np.argmax(q_table[observation]))
	observation, reward, terminated, truncated, _ = environment.step(action)
	total_reward += reward
	if terminated or truncated:
		break
environment.close()

print(f"Greedy policy reward: {total_reward}")
```
- After all 5,000 training episodes, this final block tests the **learned** policy: no more exploration, always pick the best-known action (`np.argmax`, no randomness at all) — this is called acting **greedily**.
- `total_reward` — accumulates points across the test run.
- `for _ in range(200)` — a safety limit: if the agent somehow gets stuck in a loop, stop after 200 steps regardless.
- `environment.close()` — releases the environment's internal resources cleanly (good practice, similar to closing a file after reading it).

---

### Lesson 13 — Sentiment Analysis (`13_sentiment_analysis.py`)

**New concept: turning text into numbers.** Computers can't do math on the word "excellent" directly — text must be converted to a numeric representation first, a process broadly called **feature extraction** or **vectorization**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
- **TF-IDF** (Term Frequency–Inverse Document Frequency) — converts a sentence into a row of numbers, one number per word (or word-pair) in the vocabulary. Words that appear often in *this* sentence but rarely across *all* sentences get a high score (they're considered more "distinctive" or informative); extremely common words (like "the," "a") get scored low.

```python
texts = [
	"This movie was excellent and inspiring",
	...
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
```
- 10 example movie reviews, hand-labeled: `1` = positive sentiment, `0` = negative.

```python
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(random_state=42))
```
- `ngram_range=(1, 2)` — instead of only looking at single words ("unigrams," n=1), also consider **pairs of adjacent words** ("bigrams," n=2), e.g. "not good" carries very different meaning from "good" alone — bigrams help capture that context.
- The pipeline: text → TF-IDF numeric vector → Logistic Regression classifier (the same algorithm from Lesson 09, just applied to text features instead of tumor measurements).

```python
model.fit(texts, labels)

examples = ["an enjoyable and brilliant film", "a tedious and awful movie"]
for text, sentiment in zip(examples, model.predict(examples)):
	print(f"{sentiment and 'positive' or 'negative'}: {text}")
```
- `zip(examples, model.predict(examples))` — pairs up each input sentence with its predicted label, so we can loop over both together.
- `sentiment and 'positive' or 'negative'` — a compact (if slightly cryptic) way of writing "if sentiment is truthy (i.e., 1), use 'positive', otherwise use 'negative'." (A clearer beginner-friendly version would be `'positive' if sentiment else 'negative'`.)

---

### Lesson 15 — PCA / Dimensionality Reduction (`15_pca_reduction.py`)

**New concept:** real-world datasets can have *hundreds* of features (columns). This causes problems (slow training, "the curse of dimensionality," difficulty visualizing). **Principal Component Analysis (PCA)** compresses many features down into fewer *new* features that still capture most of the original information.

```python
from sklearn.decomposition import PCA
```

```python
features, _ = load_digits(return_X_y=True)
scaled_features = StandardScaler().fit_transform(features)
```
- `return_X_y=True` — a shortcut flag on many sklearn dataset loaders: instead of returning a bundle object (`digits.data`, `digits.target`), it directly unpacks into two separate variables.
- `.fit_transform(features)` — a combo of `.fit()` (learn the mean/spread of the data) and `.transform()` (apply the scaling), done in one call since we're not doing a train/test split here.

```python
model = PCA(n_components=0.95, random_state=42)
reduced_features = model.fit_transform(scaled_features)
```
- `n_components=0.95` — an unusual but powerful way to specify PCA: instead of saying "give me exactly N new features," this says "give me the *minimum* number of new features needed to preserve 95% of the original information (variance)." PCA will decide the exact count automatically.

```python
print(f"Original dimensions: {features.shape[1]}")
print(f"Reduced dimensions: {reduced_features.shape[1]}")
print(f"Explained variance: {model.explained_variance_ratio_.sum():.3f}")
```
- `.shape[1]` — for a 2D array, `.shape` is `(rows, columns)`; `[1]` grabs just the column count (number of features).
- Digits start with 64 features (8×8 pixels); this typically compresses down to ~29 features while retaining 95%+ of the meaningful information — a huge reduction with minimal information loss.
- `.explained_variance_ratio_` — an array showing how much information each new compressed feature retained individually; `.sum()` adds them up to confirm the total (should be ≥ 0.95, matching our target).

---

### Lesson 16 — Data Preprocessing Pipeline (`16_data_preprocessing.py`)

**New concept: handling messy, mixed, incomplete real-world data** — most datasets in practice aren't clean numeric arrays; they have missing values and both numeric *and* categorical (text-label) columns.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
```
- `SimpleImputer` — fills in missing values (this dataset has a `None` for one person's age).
- `OneHotEncoder` — converts categorical text labels (like `"junior"`/`"senior"`) into numbers a model can use.
- `ColumnTransformer` — applies **different** preprocessing to **different** columns of the same dataset simultaneously.

```python
features = [[25, "junior"], [35, "senior"], [None, "junior"], [52, "senior"], [41, "senior"], [29, "junior"]]
target = [0, 1, 0, 1, 1, 0]
```
- A tiny handmade dataset: column 0 = age (numeric, with one missing value), column 1 = seniority level (categorical text).

```python
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([( "imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), [0]),
    ("category", OneHotEncoder(handle_unknown="ignore"), [1]),
])
```
- This builds two parallel "assembly lines" and glues them back together side-by-side:
  - For column `[0]` (age): first `SimpleImputer(strategy="median")` — replace the missing `None` with the **median** (middle value) of the other known ages — then `StandardScaler()` scales it.
  - For column `[1]` (seniority): `OneHotEncoder` converts `"junior"`/`"senior"` into separate binary (0/1) columns, e.g. `is_junior`, `is_senior`.
  - `handle_unknown="ignore"` — if the model later sees a category it never saw during training (e.g., `"mid-level"`), don't crash — just encode it as all-zeros.

```python
model = Pipeline([("preprocess", preprocessor), ("classifier", LogisticRegression(random_state=42))])
```
- Chains the whole preprocessing setup directly into the final classifier — one `.fit()` call handles imputation, scaling, encoding, and model training all together, in the correct order, automatically applied identically to both train and test data.

---

### Lesson 17 — Cross-Validation (`17_cross_validation.py`)

**New concept:** a single train/test split can be lucky or unlucky (what if, by chance, all the "hard" examples ended up in the test set?). **Cross-validation** solves this by testing multiple times on different slices of data and averaging the results — giving a far more trustworthy performance estimate.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
```

```python
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, features, target, cv=folds, scoring="accuracy")
```
- `StratifiedKFold(n_splits=5, ...)` — splits the full dataset into **5 equal folds** (chunks), preserving class proportions in each fold (the "stratified" part, same idea as `stratify=` from earlier lessons).
- `cross_val_score` — automatically runs the **entire** train → test cycle **5 times**: each time, one fold is held out as the test set and the other 4 are used for training, rotating through all 5 combinations. This means every data point gets used for testing exactly once.
- `shuffle=True` — randomly shuffles data before splitting into folds, so folds aren't just "the first 20% of rows, second 20%, etc." (which could be ordered/biased in real datasets).

```python
print(f"Fold accuracies: {scores.round(3)}")
print(f"Mean accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
```
- `scores` is an array of 5 accuracy numbers (one per fold).
- `.mean()` — the average performance across all 5 folds — a much more reliable estimate than a single train/test split.
- `.std()` — the **standard deviation**: how much the 5 scores varied from each other. A small `std` means the model's performance is *consistent*; a large `std` is a warning sign the model's quality depends heavily on which data it happens to see.

---

### Lesson 18 — Hyperparameter Tuning (`18_hyperparameter_tuning.py`)

**New concept: hyperparameters vs. parameters.** *Parameters* are values the model learns automatically during `.fit()` (like the line's slope in Linear Regression). *Hyperparameters* are settings **you** choose *before* training (like SVM's `kernel` type, or KNN's `n_neighbors`) — and choosing them well matters a lot. This lesson **automates** that choice.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
```
- `GridSearchCV` — "Grid Search with Cross-Validation": systematically tries **every combination** of hyperparameter values you give it, cross-validating each combination, and reports which combination performed best.

```python
model = Pipeline([("scale", StandardScaler()), ("classifier", SVC())])
search = GridSearchCV(
    model,
    {"classifier__C": [0.1, 1, 10], "classifier__kernel": ["linear", "rbf"]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy",
)
```
- The dictionary defines the "grid" to search: 3 values of `C` × 2 values of `kernel` = **6 total combinations** to try.
- `classifier__C` — the double-underscore syntax is how you target a hyperparameter *inside a named pipeline step*: "the `C` parameter, belonging to the step named `classifier`."
- `C` — SVM's regularization strength: low `C` = simpler, more tolerant boundary (may underfit); high `C` = tries harder to classify every training point correctly (may overfit — see Part 2 for full explanation of overfitting).
- Each of these 6 combinations is evaluated with full 5-fold cross-validation (from Lesson 17) — so this runs **30 total training cycles** internally.

```python
search.fit(features, target)
print(f"Best parameters: {search.best_params_}")
print(f"Best cross-validation accuracy: {search.best_score_:.3f}")
```
- `.best_params_` — the winning hyperparameter combination.
- `.best_score_` — the cross-validated accuracy that combination achieved.

---

### Lesson 19 — Gradient Boosting (`19_gradient_boosting.py`)

**New concept:** another ensemble method (like Random Forest), but with a fundamentally different strategy: instead of building many *independent* trees and voting (Random Forest), Gradient Boosting builds trees **one at a time, sequentially**, where each new tree is trained specifically to **fix the mistakes** of all the previous trees combined.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
```
- `HistGradientBoostingClassifier` — a modern, fast implementation of gradient boosting built into sklearn (conceptually similar to the popular external libraries XGBoost/LightGBM).

```python
model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42)
```
- `max_iter=150` — build 150 trees in sequence.
- `learning_rate=0.08` — how much each new tree's correction is allowed to influence the overall combined prediction. Lower values need more trees (`max_iter`) to reach the same accuracy, but tend to generalize better (less overfitting) — a classic ML tradeoff.

---

### Lesson 20 — Model Persistence (`20_model_persistence.py`)

**New concept:** training a model can take minutes, hours, or days. You don't want to retrain it every single time you want to use it — you want to **save** the trained model to a file and **load** it back later, instantly.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import joblib
```
- `joblib` — a library optimized for efficiently saving Python objects containing large NumPy arrays (which is exactly what a trained sklearn model contains internally).
- `Path` — Python's modern, cross-platform way of handling file paths (works correctly on both Windows `\` and Linux/Mac `/` path separators).
- `TemporaryDirectory` — creates a folder that automatically deletes itself when you're done (used here just so the lesson doesn't leave leftover files on your disk — in a real project you'd save to a permanent folder instead).

```python
model = LogisticRegression(max_iter=500, random_state=42).fit(features, target)
```
- Notice `.fit()` is chained directly onto the constructor in one line — a shorthand for creating and immediately training the model, since we don't need the untrained model object separately for anything else.

```python
with TemporaryDirectory() as directory:
    path = Path(directory) / "iris_model.joblib"
    joblib.dump(model, path)
    restored_model = joblib.load(path)
    print(f"Saved model: {path.name}")
    print(f"Restored model accuracy: {restored_model.score(features, target):.3f}")
```
- `with TemporaryDirectory() as directory:` — a **context manager**: guarantees the temporary folder is properly cleaned up afterward, even if an error occurs inside the block.
- `Path(directory) / "iris_model.joblib"` — the `/` operator here isn't division — `Path` objects overload it to mean "join this path with this filename," which is cleaner and safer than manually gluing strings together with `+`.
- `joblib.dump(model, path)` — serializes (converts to a saveable byte format) the entire trained model and writes it to disk at `path`.
- `joblib.load(path)` — reads that file back and reconstructs a fully working model object — `restored_model` behaves identically to the original `model`, with all the same learned parameters.
- `.score(features, target)` — a convenience method every sklearn classifier has: runs `.predict()` internally and compares to the true `target`, returning accuracy in one call (equivalent to manually calling `accuracy_score`).
- The final print confirms the restored model gets the *same* accuracy as the original — proving the save/load round-trip preserved everything correctly.

---

## Part 2 — The `tensorflow/` Track (Deep Learning)

This track uses **TensorFlow**, specifically its high-level API called **Keras**, to build **neural networks** by hand, layer by layer — giving you far more control (and far more responsibility) than scikit-learn's ready-made classifiers.

### Core Deep Learning Vocabulary (read before Lesson 01)

- **Neuron**: takes several numbers in, multiplies each by a learned **weight**, sums them plus a **bias**, and passes the result through an **activation function**.
- **Layer**: a group of neurons operating in parallel on the same input.
- **Dense layer** (`tf.keras.layers.Dense`): every neuron connects to *every* input value — the most basic, general-purpose layer type.
- **Activation function**: a nonlinear function applied after the weighted sum. Without it, stacking layers would mathematically collapse into a single equivalent layer — activations are what let networks learn complex, curved patterns. Common ones:
  - `"relu"` — outputs the input directly if positive, otherwise 0. Simple, fast, and the most common choice for hidden layers.
  - `"softmax"` — converts a layer's raw outputs into a proper probability distribution across categories (all values between 0–1, summing to 1) — used on the *final* layer for multi-class classification.
- **Loss function**: a formula measuring "how wrong was this prediction?" Training tries to minimize this number.
  - `"mse"` (Mean Squared Error) — for regression (predicting continuous numbers).
  - `"sparse_categorical_crossentropy"` — for classification, when your labels are plain integers (0, 1, 2, ...) rather than one-hot encoded vectors.
- **Optimizer**: the algorithm that actually adjusts every weight in the network, a tiny bit at a time, to reduce the loss. `"adam"` is the most popular modern default — it adapts its own step size per weight automatically.
- **Epoch**: one full pass through the entire training dataset.
- **Batch size**: instead of updating weights after *every single* example (slow, noisy) or after the *whole dataset at once* (memory-heavy, slow to start improving), training is done in small **batches** — the network sees, e.g., 64 examples, then makes one weight update, repeatedly, until it's seen the whole dataset (completing one epoch).
- **Backpropagation & Gradient Descent**: the mathematical machinery behind training — for each batch, the network calculates exactly how much each weight contributed to the error (using calculus — "gradients"), then nudges every weight slightly in the direction that reduces the error. This repeats for thousands of batches across many epochs. TensorFlow/Keras handles all of this calculus **automatically** — you never write it by hand.

### Lesson 01 — Linear Regression with a Neural Network (`01_linear_regression.py`)

This proves an important point: **a neural network with no hidden layers and no activation function is mathematically equivalent to Linear Regression.** This lesson builds the "simplest possible neural network" specifically to illustrate that connection.

```python
features = rng.uniform(-5, 5, size=(200, 1)).astype("float32")
target = (3.5 * features[:, 0] - 2 + rng.normal(0, 1, 200)).astype("float32")
```
- `rng.uniform(-5, 5, size=(200, 1))` — 200 random decimal numbers evenly spread between -5 and 5, shaped as a column (200 rows, 1 feature).
- The target is manually constructed from a known formula (`3.5 * x - 2`) **plus random noise** (`rng.normal(0, 1, 200)` — 200 random numbers from a bell-curve distribution centered at 0). This lets us later check: did the network correctly discover the "true" hidden formula (slope ≈ 3.5, intercept ≈ -2)? Real-world data never comes with a known ground-truth formula like this — it's a teaching device.

```python
model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(1,)),
	tf.keras.layers.Dense(1),
])
```
- `tf.keras.Sequential([...])` — the simplest way to build a network in Keras: a plain stack of layers, executed top to bottom.
- `tf.keras.layers.Input(shape=(1,))` — declares that each input example has exactly 1 number (matches our 1-feature data).
- `tf.keras.layers.Dense(1)` — one Dense layer with exactly **1 neuron** and **no activation function** specified (meaning: pure linear output, no squashing). One neuron here means: `output = weight * input + bias` — literally the equation of a straight line! This is *exactly* what Linear Regression computes.

```python
model.compile(optimizer="adam", loss="mse")
```
- `.compile()` — configures *how* the network will learn, before training starts: which optimizer to use, and which loss function to minimize.

```python
model.fit(features_train, target_train, epochs=250, verbose=0)
```
- `epochs=250` — the network sees the entire training set 250 times, gradually adjusting its single weight and bias to better match the line `3.5x - 2`. (Note: Adam's default learning rate of 0.001 converges slowly for a 1-weight model like this — raising it to 0.1 reaches sklearn-equivalent R² much faster.)
- `verbose=0` — suppresses Keras's normal per-epoch progress printout (which would otherwise spam 250 lines of output); `verbose=1` would show a progress bar, `verbose=2` a one-line summary per epoch.

```python
predictions = model.predict(features_test, verbose=0)[:, 0]
```
- Keras models always output a 2D array (even for a single output number per row) — shape `(40, 1)` here. `[:, 0]` extracts just that single column as a flat 1D array, matching the shape of `target_test`, so later math (subtraction, etc.) works correctly.

```python
rmse = tf.sqrt(tf.reduce_mean(tf.square(target_test - predictions)))
```
- This manually recomputes RMSE using TensorFlow's own math functions instead of scikit-learn's `mean_squared_error` — same concept as Lesson 01 of the sklearn track, just built from more basic pieces: `tf.square` (squares each error), `tf.reduce_mean` (averages them — "reduce" means "collapse an array down to a single summary number"), `tf.sqrt` (square root).

```python
target_mean = tf.reduce_mean(target_test)
r2 = 1 - tf.reduce_sum(tf.square(target_test - predictions)) / tf.reduce_sum(
	tf.square(target_test - target_mean)
)
```
- This manually computes R² from its mathematical definition, since TensorFlow doesn't have a built-in `r2_score` the way sklearn does:
  - Numerator: total squared error of *our model's* predictions.
  - Denominator: total squared error if we had just guessed the *average* target value every time (a naive baseline).
  - `1 - (our error / baseline error)` — if our model is much better than guessing the average, this ratio is small, so R² approaches 1.

```python
print(f"R2: {float(r2):.3f}")
```
- `float(r2)` — TensorFlow calculations produce special `Tensor` objects, not plain Python numbers. `float(...)` converts the result to an ordinary Python float so it can be neatly formatted and printed.

---

### Lesson 02 & most others in `tensorflow/` — Shared with sklearn

Files like `02_kmeans_clustering.py`, `04` through `09`, `13`, and `15` through `20` in the tensorflow folder are **stub files** — one-line pointers back to the sklearn implementation (see the note at the top of each: `# Shared with sklean/...`). This is because K-Means, KNN, SVM, Naive Bayes, Random Forest, Logistic Regression, sentiment analysis (TF-IDF), PCA, preprocessing pipelines, cross-validation, hyperparameter tuning, and model saving are all naturally **scikit-learn's specialty** — TensorFlow isn't the right tool for these, so the repo simply reuses the sklearn code rather than forcing an awkward TensorFlow reimplementation. Refer back to Part 1 for these.

---

### Lesson 03 — ANN on Full MNIST (`03_ann_mnist.py`, tensorflow version)

Same *concept* as sklearn's Lesson 03 (a basic neural network classifying digit images), but using the full, famous **MNIST** dataset (70,000 real handwritten digit images, 28×28 pixels each — much bigger and more realistic than sklearn's tiny 8×8 built-in version) and hand-built with Keras layers.

```python
(features_train, target_train), (features_test, target_test) = (
	tf.keras.datasets.mnist.load_data()
)
```
- Keras ships with several famous datasets built in, downloadable with one line. This automatically downloads (and caches locally for next time) the MNIST dataset, **already pre-split** into train/test — no need for `train_test_split` here.

```python
features_train = features_train[:10000].astype("float32") / 255.0
target_train = target_train[:10000]
features_test = features_test[:2000].astype("float32") / 255.0
target_test = target_test[:2000]
```
- `[:10000]` — MNIST has 60,000 training images; this **slices** it down to just the first 10,000, to make training fast enough for a quick lesson/demo on an ordinary laptop CPU. Same idea for the 2,000-image test slice.
- `/255.0` — pixel normalization, explained in Part 0.

```python
model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(28, 28)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation="relu"),
	tf.keras.layers.Dense(10, activation="softmax"),
])
```
- `Input(shape=(28, 28))` — each image is a 28×28 grid (2-dimensional), unlike Lesson 01's single flat number.
- `Flatten()` — converts that 28×28 grid into one long list of 784 numbers (28×28=784), since a plain `Dense` layer expects a flat list of inputs, not a 2D grid.
- `Dense(128, activation="relu")` — the hidden layer: 128 neurons, each looking at all 784 pixel values, learning to detect useful patterns (edges, curves, loops).
- `Dense(10, activation="softmax")` — the output layer: exactly 10 neurons (one per digit 0–9), `softmax` converts their raw scores into 10 probabilities that sum to 1 — e.g., `[0.01, 0.02, 0.90, 0.01, ...]` means "90% confident this is a 2."

```python
model.compile(
	optimizer="adam",
	loss="sparse_categorical_crossentropy",
	metrics=["accuracy"],
)
```
- `"sparse_categorical_crossentropy"` — the standard loss function for multi-class classification when labels are plain integers (`target_train` contains values like `7`, `2`, `9`, not one-hot vectors) — "sparse" refers to this integer-label format.
- `metrics=["accuracy"]` — tells Keras to also track and report accuracy (not just the raw loss number) during training, since loss values alone are hard to interpret intuitively.

```python
model.fit(features_train, target_train, epochs=3, batch_size=64, verbose=0)
_, accuracy = model.evaluate(features_test, target_test, verbose=0)
```
- `batch_size=64` — process 64 images at a time before each weight update (see Part 2 vocabulary above).
- `.evaluate(...)` — runs the model on test data and returns `[loss, accuracy]` (matching the order given in `metrics=` during `.compile()`). `_, accuracy` unpacks this list, discarding the loss value (`_`) and keeping just the accuracy.

---

### Lesson 10 — RNN for Time Series (`10_rnn_timeseries.py`)

**New concept: sequences and memory.** So far, every model treated each row of data as independent — order didn't matter. But for **time series** data (stock prices, sensor readings, sentences), the *order* and *history* of past values matters enormously. A **Recurrent Neural Network (RNN)** processes a sequence step by step, carrying forward an internal "memory" (**hidden state**) that summarizes everything seen so far.

```python
time = np.linspace(0, 40, 500, dtype=np.float32)
series = np.sin(time) + 0.1 * np.sin(3 * time)
```
- `np.linspace(0, 40, 500)` — 500 evenly-spaced numbers from 0 to 40 (like timestamps).
- `np.sin(time) + 0.1 * np.sin(3*time)` — a fake wavy signal: a big slow sine wave plus a smaller, faster "wobble" sine wave layered on top — mimicking, e.g., a seasonal trend with smaller day-to-day fluctuations.

```python
window_size = 20
features = np.array([series[index:index + window_size] for index in range(len(series) - window_size)])
target = series[window_size:]
```
- **This is the key trick for turning a time series into a supervised learning problem**, called a **sliding window**: for each position, take the previous 20 values as `features`, and the very next value as the `target` to predict.
- `series[index:index+window_size]` — a "slice" (a window) of 20 consecutive values, starting at `index`.
- The list comprehension repeats this for every possible starting position, building up hundreds of (20-values-in → 1-value-out) training examples from a single long sequence.
- `target = series[window_size:]` — the true "next value" for every window, correctly offset by 20 positions to line up.

```python
model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(window_size, 1)),
	tf.keras.layers.SimpleRNN(16),
	tf.keras.layers.Dense(1),
])
```
- `Input(shape=(window_size, 1))` — RNNs expect input shaped as `(sequence_length, features_per_timestep)`: 20 timesteps, 1 number per timestep.
- `SimpleRNN(16)` — the recurrent layer, with 16 internal "memory" units. Internally, it processes the 20 values **one at a time, in order**, updating its internal memory state at each step, and outputs a final 16-number summary vector after seeing the whole window.
- `Dense(1)` — takes that 16-number summary and produces a single predicted "next value" — same idea as Lesson 01's final Dense layer.

```python
model.fit(
	features[:split_index, :, None], target[:split_index],
	epochs=8, batch_size=32, verbose=0,
)
```
- `features[:split_index, :, None]` — `features` starts as shape `(num_windows, 20)`. The RNN needs a 3rd dimension for "features per timestep" (here, 1). `[..., None]` (or equivalently `np.newaxis`) inserts a new size-1 dimension at the end, reshaping to `(num_windows, 20, 1)` to match the `Input(shape=(window_size, 1))` declared above.

```python
loss = model.evaluate(features[split_index:, :, None], target[split_index:], verbose=0)
print(f"RNN test MSE: {loss:.4f}")
```
- Since `.compile(loss="mse")` and no `metrics=` were specified, `.evaluate()` returns just the single loss number directly (not a list, unlike Lesson 03) — so no unpacking (`_, accuracy = ...`) is needed here.

---

### Lesson 11 — Q-Learning with a Neural Network / DQN-style (`11_q_learning.py`, tensorflow version)

This solves the **exact same** CliffWalking problem as sklearn's tabular Lesson 11, but replaces the lookup-table Q-values with a small **neural network that predicts Q-values** — the foundational idea behind **Deep Q-Networks (DQN)**, the technique that famously let AI agents master Atari games.

**Why bother, if the table version already worked?** Because a table only works when there are few, distinct states (48 grid squares, here). Real-world problems (robotics, video games) can have millions or infinite possible states — you can't build a table that huge. A neural network instead *generalizes*: it can estimate a Q-value even for a state combination it has never exactly seen before, by recognizing similarity to states it has seen.

```python
model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(n_states,)),
	tf.keras.layers.Dense(32, activation="relu"),
	tf.keras.layers.Dense(n_actions),
])
```
- `Input(shape=(n_states,))` — the input is a vector as long as the total number of states (48).
- `Dense(n_actions)` — **no activation function** on the output layer. Q-values can be *any* real number (positive or negative, unbounded) — unlike classification, we don't want to squash them into a 0–1 probability range with `softmax`.
- Notice: the network takes a **whole state vector** in and outputs **all Q-values for every action at once** in a single forward pass — more efficient than running the network separately once per action.

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
```
- Here the optimizer is constructed manually (instead of just passing the string `"adam"` to `.compile()`) because this script uses a **custom training loop** (below) instead of the usual `.compile()` + `.fit()` combo — necessary because Q-learning's per-step update logic doesn't fit the standard "here's my whole dataset, fit it" pattern.

```python
def one_hot(state):
	vector = np.zeros((1, n_states), dtype="float32")
	vector[0, state] = 1.0
	return vector
```
- **One-hot encoding**: converts a single state number (e.g., state `17`) into a vector of all zeros except a single `1` at position 17 (e.g., `[0,0,...,0,1,0,...,0]`). This is the standard way to feed a "which one of N categories" input into a neural network, since the network expects a vector, not a bare integer. (Note: `environment.action_space.n` returns a NumPy integer; Keras's `Dense` layer requires a plain Python `int`, so both are wrapped in `int(...)`.)
- `np.zeros((1, n_states), ...)` — the extra `1` in the shape (`(1, n_states)` instead of just `(n_states,)`) is because Keras models always expect a **batch dimension** — even for a single example, it must be "a batch containing 1 example," not a bare unbatched vector.

```python
for episode in range(500):
	...
	while not finished:
		exploration_rate = max(0.05, 1 - episode / 400)
		state_vector = one_hot(observation)
		if random_generator.random() < exploration_rate:
			action = environment.action_space.sample()
		else:
			action = int(np.argmax(model(state_vector, training=False)[0]))
```
- Same exploration-vs-exploitation logic as the tabular version (Part 1, Lesson 11) — but instead of reading `q_table[observation]`, we now run `model(state_vector, training=False)` — a **forward pass** through the network to get its current Q-value predictions for this state.
- `training=False` — tells layers like Dropout or BatchNorm (not used here, but this is standard practice) to behave in "inference mode" rather than "training mode." Good habit even when such layers aren't present.
- `[0]` — strips off the batch dimension, since we passed in a batch of 1, getting back a plain array of `n_actions` Q-values.

```python
		next_observation, reward, terminated, truncated, _ = environment.step(action)
		next_state_vector = one_hot(next_observation)
		best_future_value = 0 if terminated else tf.reduce_max(model(next_state_vector, training=False)[0])
		target = reward + discount_factor * best_future_value
```
- Exactly the same Bellman-equation logic as before (Part 1, Lesson 11) — `target` represents "what the Q-value *should* be," combining the real reward with the network's own estimate of future value.

```python
		with tf.GradientTape() as tape:
			q_values = model(state_vector, training=True)[0]
			loss = tf.square(target - q_values[action])
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```
- **This replaces the tabular version's simple `+=` update line with actual neural network training** — this is the heart of the difference between the two approaches:
- `tf.GradientTape()` — TensorFlow's mechanism for **recording every mathematical operation** that happens inside the `with` block, specifically so it can later calculate gradients (the calculus needed for learning) through those operations automatically.
- `training=True` — this time, running in "training mode" (relevant for layers like Dropout, though again unused here — still good practice for consistency).
- `loss = tf.square(target - q_values[action])` — the squared difference between "what we think the Q-value should be" (`target`) and "what the network currently predicts for the action we actually took" (`q_values[action]`). This is the exact same TD-error concept from the tabular version, just now used as a proper loss function to train a network.
- `tape.gradient(loss, model.trainable_variables)` — computes, via calculus (**backpropagation**), exactly how much each of the network's internal weights contributed to this loss.
- `optimizer.apply_gradients(...)` — nudges every weight slightly in the direction that would have reduced this loss, using Adam's learning rate (0.01).
- `zip(gradients, model.trainable_variables)` — pairs each computed gradient with the specific weight variable it belongs to, since `apply_gradients` needs to know which gradient updates which weight.

(Note: wrapping this update logic in a function decorated with `@tf.function` compiles it into a static graph once, instead of re-tracing Python + eager TensorFlow calls on every single step — a 3–5x speedup for step-by-step training loops like this one.)

This entire block runs **once per single step** the agent takes (not once per episode, not once for the whole dataset) — this step-by-step, "learn immediately from one experience" pattern is characteristic of **online reinforcement learning**, quite different from the "gather a big dataset first, then train" pattern used everywhere else in this repo.

---

### Lesson 12 — CNN on CIFAR-10 (`12_cnn_cifar10.py`)

**New concept: Convolutional Neural Networks (CNNs)** — the standard architecture for image-related tasks. Unlike `Flatten()` + `Dense` (Lesson 03), which throws away all spatial structure (treating "pixel top-left" and "pixel bottom-right" as unrelated numbers), CNNs use small sliding filters that scan across the image, learning to detect local visual patterns (edges, textures, shapes) regardless of *where* in the image they appear.

```python
(features_train, target_train), (features_test, target_test) = (
	tf.keras.datasets.cifar10.load_data()
)
```
- **CIFAR-10**: 60,000 small, real color photographs (32×32 pixels), across 10 categories (airplane, cat, truck, etc.) — meaningfully harder than MNIST's simple grayscale digits, since real photos have much more visual complexity, and only 32×32 pixels to work with.

```python
features_train = features_train[:10000].astype("float32") / 255.0
```
- Same slicing-and-normalizing pattern as Lesson 03 — full CIFAR-10 has 50,000 training images; slicing to 10,000 keeps training time reasonable on a laptop CPU.

```python
model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(32, 32, 3)),
	tf.keras.layers.Conv2D(32, 3, activation="relu"),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Conv2D(64, 3, activation="relu"),
	tf.keras.layers.GlobalAveragePooling2D(),
	tf.keras.layers.Dense(10, activation="softmax"),
])
```
- `Input(shape=(32, 32, 3))` — CIFAR-10 images are 32×32 pixels with **3 color channels** (Red, Green, Blue) — unlike MNIST's single grayscale channel.
- `Conv2D(32, 3, activation="relu")` — a **convolutional layer**: slides `32` different learnable 3×3-pixel filters across the entire image. Each filter learns to "light up" (activate strongly) when it detects a specific small visual pattern (e.g., a vertical edge, a patch of green). The output is 32 separate "feature maps," each highlighting where in the image that particular pattern was found.
- `MaxPooling2D()` — shrinks each feature map by taking the maximum value in small 2×2 patches. This reduces the image size (fewer numbers to process further, faster computation) while keeping the strongest/most important detected signals.
- `Conv2D(64, 3, activation="relu")` — a *second* convolutional layer, with more filters (64). Because it's applied after the first `Conv2D`+`MaxPooling2D`, this layer effectively learns to detect *combinations* of the simple patterns found by the first layer — building up from edges/colors toward more complex shapes (e.g., "a rounded shape with fur texture").
- `GlobalAveragePooling2D()` — takes each of the 64 final feature maps and averages all its values down to a **single number per map**, producing one flat vector of 64 numbers. This replaces the need for a `Flatten()` + large `Dense` layer, and is a common modern technique to reduce the total number of trainable weights, which helps prevent overfitting.
- `Dense(10, activation="softmax")` — final classification layer, same idea as Lesson 03: 10 categories, `softmax` for probabilities.

```python
model.fit(features_train, target_train, epochs=20, batch_size=64, verbose=0)
```
- Note: even 20 epochs on a small 2-layer CNN with only 10,000 training images (out of the real 50,000 available) will give **moderate, not state-of-the-art** accuracy — this is intentional, it's a teaching example demonstrating the CNN architecture pattern, not a production-grade image classifier. Real CIFAR-10 systems use much deeper networks, the full dataset, data augmentation, and dozens/hundreds of epochs to reach 90%+ accuracy.

---

### Lesson 14 — Transformers / LLM Text Generation (`14_transformers_llm.py`)

**New concept:** this lesson doesn't *train* a model at all — it downloads and uses an **already-trained** (pretrained) language model, a common and highly practical real-world pattern.

```python
from transformers import pipeline
```
- The **`transformers`** library (from Hugging Face, not TensorFlow itself, though it can run on top of TensorFlow or PyTorch) provides easy access to thousands of pretrained AI models.
- `pipeline` — a high-level shortcut function: give it a task name and a model name, and it handles all the complex setup (loading the model, preparing text into the numeric format the model expects, running it, converting output back to readable text) automatically.

```python
generator = pipeline("text-generation", model="distilgpt2")
```
- `"text-generation"` — the task: given some starting text, predict/generate what comes next.
- `"distilgpt2"` — a small, fast, "distilled" (compressed) version of OpenAI's GPT-2 language model — chosen here specifically because it's lightweight enough to download and run quickly on an ordinary laptop CPU, unlike today's much larger frontier language models.
- **What is a Transformer, conceptually?** It's a neural network architecture (much more advanced than the RNN in Lesson 10) built around a mechanism called **"attention,"** which lets the model weigh the importance of *every* other word in the input when processing each word — rather than reading strictly left-to-right one step at a time like an RNN. This lets Transformers understand long-range context in text far better, and is the architecture behind essentially all modern large language models (GPT, Claude, etc.).

```python
result = generator(
	"Machine learning is",
	max_new_tokens=30,
	num_return_sequences=1,
	do_sample=False,
	clean_up_tokenization_spaces=False,
)
```
- `"Machine learning is"` — the **prompt**: the starting text the model will continue from.
- `max_new_tokens=30` — generate at most 30 new **tokens** (a token is roughly a word or word-fragment — language models don't process whole words directly, they break text into these smaller sub-word chunks).
- `num_return_sequences=1` — generate just 1 possible continuation (you could ask for several different candidate completions at once).
- `do_sample=False` — use **greedy decoding**: at each step, always pick the single most probable next token. This makes output fully deterministic (same input always gives the same output — no `random_state` needed here, unlike everywhere else in this repo). Setting this to `True` instead would introduce controlled randomness, producing more varied/creative (but less predictable) text.
- `clean_up_tokenization_spaces=False` — a minor text-formatting setting controlling whether extra spaces around punctuation get automatically cleaned up in the final output.

(Note: running this may print a harmless deprecation warning about `generation_config` and `max_length` colliding with `max_new_tokens` — this comes from the library's internal default config, not a bug in this script, and `max_new_tokens=30` still takes effect correctly.)

```python
print(result[0]["generated_text"])
```
- `pipeline(...)` returns a list of results (one per requested sequence — matching `num_return_sequences`). Each result is a dictionary; `["generated_text"]` extracts the actual text string (prompt + generated continuation combined) from the first (and only) result.

---

## Part 3 — Concepts That Appear Everywhere (Glossary)

- **Overfitting**: when a model learns the training data *too* well — including its noise and quirks — and performs great on training data but poorly on new, unseen data. Symptoms: very high training accuracy, much lower test accuracy. Ways this repo guards against it: train/test splits, cross-validation, `early_stopping=True`, regularization (SVM's `C`), and limiting model complexity.
- **Underfitting**: the opposite problem — the model is too simple to capture the real pattern, and performs poorly on *both* training and test data.
- **Pipeline** (`make_pipeline`/`Pipeline`): chaining preprocessing steps and a model together so they always execute identically and in the correct order, on any new data — avoiding a very common real-world bug where someone forgets to scale/encode new data the exact same way training data was processed.
- **`.fit()` / `.predict()` / `.transform()`** — the three core scikit-learn method names you'll see everywhere: `fit` = learn from data, `predict` = output a guess for new data, `transform` = apply a learned transformation (like scaling) to new data without making a prediction.
- **Metrics recap:**
  - **Regression**: R² (higher/closer-to-1 is better), RMSE (lower is better, in original units).
  - **Classification**: Accuracy (simple, but misleading if classes are imbalanced), Precision/Recall/F1 (better for imbalanced or high-stakes classes), Silhouette score (for clustering only).

---

## Suggested Reading Order for a True Beginner

1. Part 0 (this whole document's foundation)
2. sklearn Lessons 01 → 04 → 05 → 09 (regression → split → KNN → logistic regression — the core supervised-learning loop)
3. sklearn Lessons 06 → 07 → 08 → 19 (other classifiers, to see the variety of approaches)
4. sklearn Lessons 02 → 15 (unsupervised learning: clustering, dimensionality reduction)
5. sklearn Lessons 16 → 17 → 18 → 20 (the "software engineering" side of ML: pipelines, validation, tuning, saving)
6. sklearn/tensorflow Lesson 03 (bridge into neural networks)
7. tensorflow Part 2 vocabulary section, then Lessons 01 → 03 → 12 → 10 (regression → basic ANN → CNN → RNN, in increasing complexity)
8. sklearn/tensorflow Lesson 11 (reinforcement learning — a genuinely different paradigm, best understood last)
9. tensorflow Lesson 14 (pretrained Transformers — ties everything together with the most modern, practical technique)