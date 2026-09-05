# ML Lab

A compact, executable overview of machine learning and artificial neural networks. Every lesson is a black-box implementation built around established Python libraries, with deterministic seeds where randomness is involved.

Most topics share identical code between tracks; `10`, `12`, and `14` are TensorFlow-only (the sklearn file points to the tensorflow one). `11` differs: sklearn uses tabular Q-learning, tensorflow uses a neural Q-function (DQN-style) trained with `tf.GradientTape`.

## Setup

Use Python 3.13 for TensorFlow on native Windows:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

In Git Bash, activate with:

```bash
source .venv/Scripts/activate
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

Run every lesson in one track from Git Bash:

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

The first execution of MNIST, CIFAR-10, and the transformer lesson downloads and caches data or model files. Native Windows TensorFlow uses the CPU; GPU workloads are better suited to WSL2 or a Linux environment. Generated caches, environments, and local model/output artifacts are excluded by `.gitignore`.