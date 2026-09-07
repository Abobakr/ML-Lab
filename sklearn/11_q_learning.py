import numpy as np
import gymnasium as gym


environment = gym.make("CliffWalking-v1")
random_generator = np.random.default_rng(42)
q_table = np.zeros((environment.observation_space.n, environment.action_space.n))
learning_rate = 0.1
discount_factor = 0.95

for episode in range(5000):
	observation, _ = environment.reset(seed=episode)
	finished = False
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
		finished = terminated or truncated

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