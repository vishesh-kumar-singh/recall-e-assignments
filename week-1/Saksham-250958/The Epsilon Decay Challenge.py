import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Q-Learning Parameters

ALPHA = 0.1          # Learning rate
GAMMA = 0.99         # Discount factor
EPISODES = 500
MAX_STEPS = 100


# Environment

env = gym.make("CliffWalking-v1")
# If taking CliffWalking-v0 then the code is not running saying it is outdated, so take CliffWalking-v1 instead.
n_states = env.observation_space.n
n_actions = env.action_space.n



# Epsilon Strategies


def constant_high(_):
    return 0.5


def constant_low(_):
    return 0.05


def decaying_epsilon(episode):
    start_epsilon = 1.0
    end_epsilon = 0.01

    decay_rate = (start_epsilon - end_epsilon) / EPISODES

    epsilon = start_epsilon - decay_rate * episode

    return max(end_epsilon, epsilon)



# Epsilon-Greedy Action Selection


def choose_action(state, q_table, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])



# Q-Learning Training Function


def train_agent(strategy_function, strategy_name):

    q_table = np.zeros((n_states, n_actions))

    rewards_per_episode = []

    for episode in range(EPISODES):

        state, _ = env.reset()

        total_reward = 0

        epsilon = strategy_function(episode)

        for step in range(MAX_STEPS):

            # Select action
            action = choose_action(state, q_table, epsilon)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Q-Learning Update
            old_value = q_table[state, action]

            next_max = np.max(q_table[next_state])

            new_value = old_value + ALPHA * (
                reward + GAMMA * next_max - old_value
            )

            q_table[state, action] = new_value

            state = next_state

            total_reward += reward

            if terminated or truncated:
                break

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"{strategy_name} -> Episode {episode + 1} completed")

    return q_table, rewards_per_episode



# Train All Agents

print("Training agent with constant high exploration...")
q_high, rewards_high = train_agent(constant_high, "High Exploration")

print("\nTraining agent with constant low exploration...")
q_low, rewards_low = train_agent(constant_low, "Low Exploration")

print("\nTraining agent with decaying exploration...")
q_decay, rewards_decay = train_agent(decaying_epsilon, "Decaying Exploration")



# Plot Results

plt.figure(figsize=(12, 6))

plt.plot(rewards_high, label="Constant High Exploration (epsilon=0.5)")
plt.plot(rewards_low, label="Constant Low Exploration (epsilon=0.05)")
plt.plot(rewards_decay, label="Decaying Exploration")

plt.xlabel("Episodes")
plt.ylabel("Sum of Rewards")
plt.title("Q-Learning on CliffWalking-v1")
plt.legend()
plt.grid(True)

plt.show()



env.close()
