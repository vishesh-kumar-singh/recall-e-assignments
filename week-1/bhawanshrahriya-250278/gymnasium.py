import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

num_episodes = 500
alpha = 0.1
gamma = 0.99

env = gym.make("CliffWalking-v1")
num_states = env.observation_space.n
num_actions = env.action_space.n

q_table_high = np.zeros((num_states, num_actions))
rewards_high = []
epsilon_high = 0.5

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    truncated = False
    total_reward = 0

    while not (done or truncated):
        if np.random.rand() < epsilon_high:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table_high[state])

        next_state, reward, done, truncated, info = env.step(action)

        best_next_action = np.argmax(q_table_high[next_state])
        td_target = reward + gamma * q_table_high[next_state][best_next_action]
        q_table_high[state][action] = q_table_high[state][action] + alpha * (td_target - q_table_high[state][action])

        state = next_state
        total_reward += reward

    rewards_high.append(total_reward)

q_table_low = np.zeros((num_states, num_actions))
rewards_low = []
epsilon_low = 0.05

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    truncated = False
    total_reward = 0

    while not (done or truncated):
        if np.random.rand() < epsilon_low:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table_low[state])

        next_state, reward, done, truncated, info = env.step(action)

        best_next_action = np.argmax(q_table_low[next_state])
        td_target = reward + gamma * q_table_low[next_state][best_next_action]
        q_table_low[state][action] = q_table_low[state][action] + alpha * (td_target - q_table_low[state][action])

        state = next_state
        total_reward += reward

    rewards_low.append(total_reward)

q_table_decay = np.zeros((num_states, num_actions))
rewards_decay = []
epsilon_decay = 1.0

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    truncated = False
    total_reward = 0

    epsilon_decay = max(0.01, 1.0 - (episode / (num_episodes * 0.8)))

    while not (done or truncated):
        if np.random.rand() < epsilon_decay:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table_decay[state])

        next_state, reward, done, truncated, info = env.step(action)

        best_next_action = np.argmax(q_table_decay[next_state])
        td_target = reward + gamma * q_table_decay[next_state][best_next_action]
        q_table_decay[state][action] = q_table_decay[state][action] + alpha * (td_target - q_table_decay[state][action])

        state = next_state
        total_reward += reward

    rewards_decay.append(total_reward)

env.close()

smoothed_high = []
smoothed_low = []
smoothed_decay = []

for i in range(len(rewards_high) - 10):
    smoothed_high.append(np.mean(rewards_high[i:i+10]))
    smoothed_low.append(np.mean(rewards_low[i:i+10]))
    smoothed_decay.append(np.mean(rewards_decay[i:i+10]))

plt.figure(figsize=(10, 6))
plt.plot(smoothed_high, label="Constant High (0.5)")
plt.plot(smoothed_low, label="Constant Low (0.05)")
plt.plot(smoothed_decay, label="Decaying (1.0 to 0.01)")

plt.title("Q-Learning Performance Comparison")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.ylim(-150, 0)
plt.legend()
plt.grid(True)
plt.show()