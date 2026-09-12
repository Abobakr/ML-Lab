import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay


tf.random.set_seed(42)
(features_train, target_train), (features_test, target_test) = (
	tf.keras.datasets.mnist.load_data()
)
features_train = features_train[:10000].astype("float32") / 255.0
target_train = target_train[:10000]
features_test = features_test[:2000].astype("float32") / 255.0
target_test = target_test[:2000]

model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(28, 28)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation="relu"),
	tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(
	optimizer="adam",
	loss="sparse_categorical_crossentropy",
	metrics=["accuracy"],
)
history = model.fit(features_train, target_train, epochs=3, batch_size=64, verbose=0)
_, accuracy = model.evaluate(features_test, target_test, verbose=0)
predictions = model.predict(features_test, verbose=0).argmax(axis=1)

print(f"ANN MNIST accuracy: {accuracy:.3f}")

# Visualization: learning curves, confusion, and sample predictions show progress and errors.
figure, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(history.history["accuracy"], label="Accuracy")
axes[0].plot(history.history["loss"], label="Loss")
axes[0].set(title="MNIST training history", xlabel="Epoch")
axes[0].legend()
ConfusionMatrixDisplay.from_predictions(target_test, predictions, ax=axes[1], cmap="Blues")
axes[1].set(title="MNIST confusion matrix")
axes[2].imshow(features_test[0], cmap="gray")
axes[2].set(title=f"Example: predicted {predictions[0]}", xticks=[], yticks=[])
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
