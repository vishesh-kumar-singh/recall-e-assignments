import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create environment
env = gym.make("CliffWalking-v1")

# Hyperparameters
alpha = 0.1
gamma = 0.99
episodes = 10000

# Environment info
n_states = env.observation_space.n
n_actions = env.action_space.n


def train_agent(strategy):

    q_table = np.zeros((n_states, n_actions))
    rewards = []

    epsilon = 1.0

    for episode in range(episodes):

        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        # Exploration strategies
        if strategy == "high":
            epsilon = 0.5

        elif strategy == "low":
            epsilon = 0.05

        elif strategy == "decay":
            epsilon = max(0.01, epsilon * 0.995)

        while not done and step_count < 100:

            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # Q-learning update
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            q_table[state, action] = old_value + alpha * (
                reward + gamma * next_max - old_value
            )

            state = next_state

            total_reward += reward

            # Reward clipping
            if total_reward < -1000:
                total_reward = -1000

            step_count += 1

        rewards.append(total_reward)

    return rewards


# Moving average smoothing
def moving_average(data, window_size=200):
    return np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode='valid'
    )


# Train agents
high_rewards = train_agent("high")
low_rewards = train_agent("low")
decay_rewards = train_agent("decay")

# Smooth rewards
high_smooth = moving_average(high_rewards)
low_smooth = moving_average(low_rewards)
decay_smooth = moving_average(decay_rewards)

# Plot graph
plt.figure(figsize=(14, 7))

plt.plot(
    high_smooth,
    label="Constant High Exploration (ε=0.5)"
)

plt.plot(
    low_smooth,
    label="Constant Low Exploration (ε=0.05)"
)

plt.plot(
    decay_smooth,
    label="Decaying Exploration"
)

plt.title("Q-Learning Performance")
plt.xlabel("Episode")
plt.ylabel("Sum of Rewards")

plt.legend()
plt.grid(True)

plt.show()