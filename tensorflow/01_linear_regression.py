import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path


tf.random.set_seed(42)
rng = np.random.default_rng(42)
features = rng.uniform(-5, 5, size=(200, 1)).astype("float32")
target = (3.5 * features[:, 0] - 2 + rng.normal(0, 1, 200)).astype("float32")
split_index = 160
features_train, features_test = features[:split_index], features[split_index:]
target_train, target_test = target[:split_index], target[split_index:]

model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(1,)),
	tf.keras.layers.Dense(1),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mse")
model.fit(features_train, target_train, epochs=250, verbose=0)
predictions = model.predict(features_test, verbose=0)[:, 0]
rmse = tf.sqrt(tf.reduce_mean(tf.square(target_test - predictions)))
target_mean = tf.reduce_mean(target_test)
r2 = 1 - tf.reduce_sum(tf.square(target_test - predictions)) / tf.reduce_sum(
	tf.square(target_test - target_mean)
)

print(f"R2: {float(r2):.3f}")
print(f"RMSE: {float(rmse):.3f}")

# Visualization: show the held-out data and the network's learned line.
order = np.argsort(features_test[:, 0])
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(features_train[:, 0], target_train, alpha=0.35, label="Train")
axes[0].scatter(features_test[:, 0], target_test, alpha=0.8, label="Test")
axes[0].plot(features_test[order, 0], predictions[order], color="black", label="Prediction")
axes[0].set(title="Learned regression line", xlabel="Feature", ylabel="Target")
axes[0].legend()
axes[1].scatter(predictions, target_test - predictions, alpha=0.8)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set(title="Residuals", xlabel="Predicted target", ylabel="Residual")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
