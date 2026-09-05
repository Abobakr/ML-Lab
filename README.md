# ML Lab

A compact, executable overview of machine learning and artificial neural networks. Every lesson is a black-box implementation built around established Python libraries, with deterministic seeds where randomness is involved.

Most topics share identical code between tracks; `10`, `12`, and `14` are TensorFlow-only (the sklearn file points to the tensorflow one). `11` differs: sklearn uses tabular Q-learning, tensorflow uses a neural Q-function (DQN-style) trained with `tf.GradientTape`.

## Setup (WSL / Linux)

Set up Python 3.13 and create a virtual environment in WSL:

```bash
# Add deadsnakes PPA and install Python 3.13
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update && sudo apt install python3.13 python3.13-venv python3.13-dev -y

# Create virtual environment
python3.13 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Tracks

| Folder | Focus |
| --- | --- |
| `sklearn/` | Classical ML, evaluation, preprocessing, tuning, and persistence |
| `tensorflow/` | Keras neural networks plus appropriate classical-library examples |

Both tracks contain the same 20 numbered topics. Run a lesson from the project root, for example:

```bash
python sklearn/18_hyperparameter_tuning.py
python tensorflow/03_ann_mnist.py
```

Run every lesson in one track directly from your terminal:

```bash
for folder in sklearn tensorflow; do
  for file in "$folder"/*.py; do
    echo "Running $file"
    python "$file" || break 2
  done
done
```

## Curriculum

`01` Linear regression, `02` K-Means, `03` ANN/MNIST, `04` train/test split, `05` KNN, `06` SVM, `07` Naive Bayes, `08` random forest, `09` logistic regression, `10` RNN/time series, `11` Q-learning, `12` CNN/CIFAR-10, `13` sentiment analysis, `14` transformers, `15` PCA, `16` preprocessing, `17` cross-validation, `18` hyperparameter tuning, `19` gradient boosting, and `20` model persistence.

## Notes

The first execution of MNIST, CIFAR-10, and the transformer lesson downloads and caches data or model files. Running inside WSL2 enables native Linux TensorFlow. GPU acceleration additionally requires an NVIDIA driver on Windows and matching CUDA/cuDNN packages inside WSL — it is not automatic. Generated caches, environments, and local model/output artifacts are excluded by `.gitignore`.

`tensorflow/11_q_learning.py` casts `environment.observation_space.n` / `.action_space.n` to `int()` — Gymnasium returns NumPy integers, which newer Keras versions reject for layer `units`.