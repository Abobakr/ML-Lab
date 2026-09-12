import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import gymnasium as gym


environment = gym.make("CliffWalking-v1")
tf.random.set_seed(42)
random_generator = np.random.default_rng(42)
n_states = int(environment.observation_space.n)
n_actions = int(environment.action_space.n)

model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(n_states,)),
	tf.keras.layers.Dense(32, activation="relu"),
	tf.keras.layers.Dense(n_actions),
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
discount_factor = 0.95
episode_rewards = []
episode_losses = []


def one_hot(state):
	vector = np.zeros((1, n_states), dtype="float32")
	vector[0, state] = 1.0
	return vector


@tf.function
def train_step(state_vector, action, target):
	with tf.GradientTape() as tape:
		q_values = model(state_vector, training=True)[0]
		loss = tf.square(target - q_values[action])
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	return loss


for episode in range(500):
	observation, _ = environment.reset(seed=episode)
	finished = False
	step_count = 0
	episode_reward = 0
	episode_loss = []
	while not finished and step_count < 100:
		exploration_rate = max(0.05, 1 - episode / 400)
		state_vector = one_hot(observation)
		if random_generator.random() < exploration_rate:
			action = environment.action_space.sample()
		else:
			action = int(np.argmax(model(state_vector, training=False)[0]))

		next_observation, reward, terminated, truncated, _ = environment.step(action)
		next_state_vector = one_hot(next_observation)
		best_future_value = 0.0 if terminated else float(tf.reduce_max(model(next_state_vector, training=False)[0]))
		target = tf.constant(reward + discount_factor * best_future_value, dtype="float32")

		loss = train_step(tf.constant(state_vector), tf.constant(action), target)
		episode_loss.append(float(loss))

		observation = next_observation
		episode_reward += reward
		finished = terminated or truncated
		step_count += 1
	episode_rewards.append(episode_reward)
	episode_losses.append(np.mean(episode_loss))

observation, _ = environment.reset(seed=123)
total_reward = 0
for _ in range(200):
	action = int(np.argmax(model(one_hot(observation), training=False)[0]))
	observation, reward, terminated, truncated, _ = environment.step(action)
	total_reward += reward
	if terminated or truncated:
		break
environment.close()

print(f"Greedy policy reward: {total_reward}")

# Visualization: unlike tabular Q-learning, the neural agent also has a loss curve.
figure, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(np.convolve(episode_rewards, np.ones(25) / 25, mode="valid"))
axes[0].set(title="DQN reward trend", xlabel="Episode", ylabel="25-episode mean reward")
axes[1].plot(episode_losses)
axes[1].set(title="DQN training loss", xlabel="Episode", ylabel="Mean loss")
policy = np.argmax(model(np.eye(n_states, dtype="float32"), training=False).numpy(), axis=1).reshape(4, 12)
axes[2].imshow(policy, cmap="viridis")
axes[2].set(title="Greedy neural policy", xlabel="Grid column", ylabel="Grid row")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)