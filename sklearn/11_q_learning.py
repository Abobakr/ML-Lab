import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import gymnasium as gym


environment = gym.make("CliffWalking-v1")
random_generator = np.random.default_rng(42)
q_table = np.zeros((environment.observation_space.n, environment.action_space.n))
learning_rate = 0.1
discount_factor = 0.95
episode_rewards = []

for episode in range(5000):
	observation, _ = environment.reset(seed=episode)
	finished = False
	episode_reward = 0
	while not finished:
		exploration_rate = max(0.05, 1 - episode / 4000)
		if random_generator.random() < exploration_rate:
			action = environment.action_space.sample()
		else:
			action = int(np.argmax(q_table[observation]))
		next_observation, reward, terminated, truncated, _ = environment.step(action)
		best_future_value = 0 if terminated else np.max(q_table[next_observation])
		q_table[observation, action] += learning_rate * (
			reward + discount_factor * best_future_value - q_table[observation, action]
		)
		observation = next_observation
		episode_reward += reward
		finished = terminated or truncated
	episode_rewards.append(episode_reward)

observation, _ = environment.reset(seed=123)
total_reward = 0
for _ in range(200):
	action = int(np.argmax(q_table[observation]))
	observation, reward, terminated, truncated, _ = environment.step(action)
	total_reward += reward
	if terminated or truncated:
		break
environment.close()

print(f"Greedy policy reward: {total_reward}")

# Visualization: rewards and greedy actions expose the learned tabular policy.
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(np.convolve(episode_rewards, np.ones(100) / 100, mode="valid"))
axes[0].set(title="Q-learning reward trend", xlabel="Episode", ylabel="100-episode mean reward")
policy = np.argmax(q_table, axis=1).reshape(4, 12)
axes[1].imshow(policy, cmap="viridis")
axes[1].set(title="Greedy policy", xlabel="Grid column", ylabel="Grid row")
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)