import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


def run_q_learning(eps_mode, episodes=500):
    env = gym.make('CliffWalking-v1')
    q_table = np.zeros((48, 4))
    rewards = []

    alpha = 0.1
    gamma = 0.99

    for i in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        # Set epsilon based on mode
        if eps_mode == 'high':
            eps = 0.5
        elif eps_mode == 'low':
            eps = 0.05
        elif eps_mode == 'decay':
            eps = max(0.05, 1.0 * (0.99 ** i))

        while not done:
            # Choose action
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Take step
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            # Update Q-table
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state, best_next_action]
            q_table[state, action] += alpha * \
                (td_target - q_table[state, action])

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    return rewards


# Run training
rewards_high = run_q_learning('high')
rewards_low = run_q_learning('low')
rewards_decay = run_q_learning('decay')

# Plot results
plt.plot(rewards_high, label='High (0.5)', alpha=0.7)
plt.plot(rewards_low, label='Low (0.05)', alpha=0.7)
plt.plot(rewards_decay, label='Decay', alpha=0.9)
plt.ylim(-200, 0)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.legend()
plt.savefig('plot.png')
