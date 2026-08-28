import tensorflow as tf


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
model.fit(features_train, target_train, epochs=3, batch_size=64, verbose=0)
_, accuracy = model.evaluate(features_test, target_test, verbose=0)

print(f"ANN MNIST accuracy: {accuracy:.3f}")
