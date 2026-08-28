import numpy as np
import tensorflow as tf


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
model.compile(optimizer="adam", loss="mse")
model.fit(features_train, target_train, epochs=250, verbose=0)
predictions = model.predict(features_test, verbose=0)[:, 0]
rmse = tf.sqrt(tf.reduce_mean(tf.square(target_test - predictions)))
target_mean = tf.reduce_mean(target_test)
r2 = 1 - tf.reduce_sum(tf.square(target_test - predictions)) / tf.reduce_sum(
	tf.square(target_test - target_mean)
)

print(f"R2: {float(r2):.3f}")
print(f"RMSE: {float(rmse):.3f}")
