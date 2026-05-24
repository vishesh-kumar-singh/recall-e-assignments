import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
alpha = 0.1
gamma = 0.99
episodes = 500

# Create Cliff Walking environment
env = gym.make("CliffWalking-v1")

# Number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Epsilon-greedy action selection
def epsilon_greedy(Q, state, epsilon):

    if np.random.rand() < epsilon:
        return env.action_space.sample()

    return np.argmax(Q[state])

# Training function
def train_agent(strategy="high"):

    # Initialize Q-table
    Q = np.zeros((n_states, n_actions))

    rewards_per_episode = []

    for episode in range(episodes):

        state, _ = env.reset()

        done = False

        total_reward = 0

        # Exploration strategies
        if strategy == "high":
            epsilon = 0.5

        elif strategy == "low":
            epsilon = 0.1

        elif strategy == "decay":
            epsilon = max(0.01, 1.0 * (0.995 ** episode))

        while not done:

            # Choose action
            action = epsilon_greedy(Q, state, epsilon)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # Q-learning update
            Q[state, action] = Q[state, action] + alpha * (
                reward
                + gamma * np.max(Q[next_state])
                - Q[state, action]
            )

            state = next_state

            total_reward += reward

        rewards_per_episode.append(total_reward)

    return rewards_per_episode

# Train three agents
high_rewards = train_agent("high")

low_rewards = train_agent("low")

decay_rewards = train_agent("decay")

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(high_rewards, label="Constant High Exploration (ε = 0.5)")

plt.plot(low_rewards, label="Constant Low Exploration (ε = 0.1)")

plt.plot(decay_rewards, label="Decaying Exploration")

plt.xlabel("Episodes")

plt.ylabel("Total Reward")

plt.title("Q-Learning on CliffWalking-v0")

plt.legend()

plt.grid()

# Save graph
plt.savefig("rewards_plot.png")

plt.show()