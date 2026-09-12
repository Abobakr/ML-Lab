import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path


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
predictions = model.predict(features[split_index:, :, None], verbose=0)[:, 0]

print(f"RNN test MSE: {loss:.4f}")

# Visualization: held-out forecasts should follow the time-series pattern.
figure, axis = plt.subplots(figsize=(10, 4))
test_time = time[window_size + split_index:]
axis.plot(test_time, target[split_index:], label="Actual")
axis.plot(test_time, predictions, label="Predicted")
axis.set(title="RNN time-series forecast", xlabel="Time", ylabel="Value")
axis.legend()
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
