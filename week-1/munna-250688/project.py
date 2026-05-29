# =========================================================
# Q-Learning on CliffWalking-v1 using Gymnasium
# =========================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Create Environment
# ---------------------------------------------------------
env = gym.make("CliffWalking-v1")

# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
episodes = 500       # Number of training episodes
max_steps = 100      # Max steps per episode

# ---------------------------------------------------------
# Q-Learning Function
# ---------------------------------------------------------
def q_learning(strategy):

    # Create Q-table
    q_table = np.zeros(
        (env.observation_space.n, env.action_space.n)
    )

    rewards_per_episode = []

    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------
    for episode in range(episodes):

        # Reset environment
        state, info = env.reset()

        total_reward = 0

        # -------------------------------------------------
        # Exploration Strategy
        # -------------------------------------------------
        if strategy == "high":
            epsilon = 0.9

        elif strategy == "low":
            epsilon = 0.1

        elif strategy == "decay":

            # Exponential decay
            epsilon = max(
                0.01,
                1.0 * (0.995 ** episode)
            )

        # -------------------------------------------------
        # Episode Loop
        # -------------------------------------------------
        for step in range(max_steps):

            # Epsilon-greedy action selection
            random_number = np.random.random()

            if random_number < epsilon:

                # Explore
                action = env.action_space.sample()

            else:

                # Exploit
                action = np.argmax(q_table[state])

            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)

            # -------------------------------------------------
            # Q-Learning Update Rule
            # -------------------------------------------------
            old_value = q_table[state, action]

            next_max = np.max(q_table[next_state])

            new_value = old_value + alpha * (
                reward
                + gamma * next_max
                - old_value
            )

            q_table[state, action] = new_value

            # Move to next state
            state = next_state

            # Add reward
            total_reward += reward

            # End episode if done
            if terminated or truncated:
                break

        # Store reward
        rewards_per_episode.append(total_reward)

    return rewards_per_episode, q_table


# ---------------------------------------------------------
# Train Three Agents
# ---------------------------------------------------------
print("Training High Exploration Agent...")
high_rewards, high_q = q_learning("high")

print("Training Low Exploration Agent...")
low_rewards, low_q = q_learning("low")

print("Training Decaying Exploration Agent...")
decay_rewards, decay_q = q_learning("decay")


# ---------------------------------------------------------
# Plot Results
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(
    high_rewards,
    label="High Exploration (ε = 0.9)"
)

plt.plot(
    low_rewards,
    label="Low Exploration (ε = 0.1)"
)

plt.plot(
    decay_rewards,
    label="Decaying Exploration"
)

# Labels
plt.xlabel("Episodes")
plt.ylabel("Sum of Rewards")

# Title
plt.title("Q-Learning on CliffWalking-v1")

# Legend
plt.legend()

# Grid
plt.grid(True)

# Show plot
plt.show()


# ---------------------------------------------------------
# Final Analysis
# ---------------------------------------------------------
print("\n================ FINAL ANALYSIS ================\n")

print("1. Constant High Exploration (ε = 0.9)")
print("- Explores too much")
print("- Falls into cliff frequently")
print("- Learning is slow and unstable\n")

print("2. Constant Low Exploration (ε = 0.1)")
print("- Learns safe path quickly")
print("- More stable rewards")
print("- May miss optimal path\n")

print("3. Decaying Exploration")
print("- Explores early and exploits later")
print("- Usually achieves best long-term performance")
print("- Learns near-optimal path\n")

print("Conclusion:")
print("Low exploration learns the safe path fastest.")
print("Decaying exploration usually finds the best overall policy.")