import tensorflow as tf


tf.random.set_seed(42)
(features_train, target_train), (features_test, target_test) = (
	tf.keras.datasets.cifar10.load_data()
)
features_train = features_train[:10000].astype("float32") / 255.0
target_train = target_train[:10000]
features_test = features_test[:2000].astype("float32") / 255.0
target_test = target_test[:2000]

model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(32, 32, 3)),
	tf.keras.layers.Conv2D(32, 3, activation="relu"),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Conv2D(64, 3, activation="relu"),
	tf.keras.layers.GlobalAveragePooling2D(),
	tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(features_train, target_train, epochs=20, batch_size=64, verbose=0)
_, accuracy = model.evaluate(features_test, target_test, verbose=0)

print(f"CNN CIFAR-10 accuracy: {accuracy:.3f}")
