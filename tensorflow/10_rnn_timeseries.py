import numpy as np
import tensorflow as tf


tf.random.set_seed(42)
time = np.linspace(0, 40, 500, dtype=np.float32)
series = np.sin(time) + 0.1 * np.sin(3 * time)
window_size = 20
features = np.array([series[index:index + window_size] for index in range(len(series) - window_size)])
target = series[window_size:]
split_index = int(len(features) * 0.8)

model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(window_size, 1)),
	tf.keras.layers.SimpleRNN(16),
	tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(
	features[:split_index, :, None], target[:split_index],
	epochs=8, batch_size=32, verbose=0,
)
loss = model.evaluate(features[split_index:, :, None], target[split_index:], verbose=0)

print(f"RNN test MSE: {loss:.4f}")
