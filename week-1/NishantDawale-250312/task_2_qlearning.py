"""
Task 2: Epsilon-Decay Challenge
Q-Learning on CliffWalking-v0

This script implements tabular Q-learning with three exploration strategies:
  1. Constant High Epsilon  (epsilon = 0.5)
  2. Constant Low Epsilon   (epsilon = 0.05)
  3. Decaying Epsilon       (1.0 → 0.1, linear decay)

The CliffWalking environment is implemented from scratch following the
standard Gymnasium CliffWalking-v0 specification, so no external RL
library is required.  If you have Gymnasium installed, you can swap in:
    import gymnasium as gym
    env = gym.make("CliffWalking-v0")
and the rest of the code works unchanged.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Self-contained CliffWalking Environment  (matches Gymnasium CliffWalking-v0)
# ─────────────────────────────────────────────────────────────────────────────

class CliffWalkingEnv:
    """
    4×12 grid world.
    - Start:  bottom-left  (row=3, col=0)  → state 36
    - Goal:   bottom-right (row=3, col=11) → state 47
    - Cliff:  bottom row, col 1-10          → states 37-46
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    Step reward: -1 everywhere, -100 on cliff (episode resets to start).
    """
    ROWS, COLS = 4, 12

    def __init__(self):
        self.n_states  = self.ROWS * self.COLS   # 48
        self.n_actions = 4
        self._start    = 36
        self._goal     = 47
        self._cliff    = set(range(37, 47))
        self.state     = self._start

    # Encode/decode state ↔ (row, col)
    def _encode(self, row, col): return row * self.COLS + col
    def _decode(self, s):        return divmod(s, self.COLS)

    def reset(self):
        self.state = self._start
        return self.state

    def step(self, action):
        row, col = self._decode(self.state)
        # Move
        if   action == 0: row = max(row - 1, 0)               # Up
        elif action == 1: col = min(col + 1, self.COLS - 1)   # Right
        elif action == 2: row = min(row + 1, self.ROWS - 1)   # Down
        elif action == 3: col = max(col - 1, 0)               # Left

        next_state = self._encode(row, col)

        if next_state in self._cliff:
            reward = -100
            next_state = self._start      # reset to start
            terminated = False
        elif next_state == self._goal:
            reward = -1
            terminated = True
        else:
            reward = -1
            terminated = False

        self.state = next_state
        return next_state, reward, terminated


# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning Agent
# ─────────────────────────────────────────────────────────────────────────────

def q_learning(env, n_episodes=500, alpha=0.5, gamma=0.99,
               epsilon_mode='constant_high', seed=42):
    """
    Train a Q-learning agent.

    Parameters
    ----------
    env          : environment instance
    n_episodes   : number of training episodes
    alpha        : learning rate
    gamma        : discount factor
    epsilon_mode : 'constant_high' | 'constant_low' | 'decaying'
    seed         : random seed for reproducibility

    Returns
    -------
    rewards_per_episode : list of total reward per episode
    Q                   : learned Q-table
    """
    np.random.seed(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode = []

    # Epsilon schedule
    if epsilon_mode == 'constant_high':
        epsilons = [0.5] * n_episodes
    elif epsilon_mode == 'constant_low':
        epsilons = [0.05] * n_episodes
    else:  # decaying
        eps_start, eps_end = 1.0, 0.1
        epsilons = np.linspace(eps_start, eps_end, n_episodes).tolist()

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        epsilon = epsilons[episode]

        for _ in range(500):   # cap steps per episode
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q-learning update (off-policy TD)
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            if done:
                break

        rewards_per_episode.append(total_reward)

    return rewards_per_episode, Q


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing helper
# ─────────────────────────────────────────────────────────────────────────────

def smooth(data, window=20):
    """Simple moving average for cleaner plots."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


# ─────────────────────────────────────────────────────────────────────────────
# Main: Train all three agents
# ─────────────────────────────────────────────────────────────────────────────

N_EPISODES = 500
ALPHA      = 0.5
GAMMA      = 0.99

print("Training Agent 1: Constant High Epsilon (ε = 0.50)...")
env1 = CliffWalkingEnv()
rewards_high, Q_high = q_learning(env1, N_EPISODES, ALPHA, GAMMA,
                                   epsilon_mode='constant_high', seed=0)

print("Training Agent 2: Constant Low Epsilon  (ε = 0.05)...")
env2 = CliffWalkingEnv()
rewards_low, Q_low = q_learning(env2, N_EPISODES, ALPHA, GAMMA,
                                 epsilon_mode='constant_low',  seed=1)

print("Training Agent 3: Decaying Epsilon  (1.0 → 0.1 linear)...")
env3 = CliffWalkingEnv()
rewards_decay, Q_decay = q_learning(env3, N_EPISODES, ALPHA, GAMMA,
                                     epsilon_mode='decaying',      seed=2)

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

WINDOW = 20
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CliffWalking Q-Learning: Exploration Strategy Comparison",
             fontsize=14, fontweight='bold', y=1.01)

colors_map = {
    'high':  ('#e74c3c', 'Constant High ε=0.50'),
    'low':   ('#2ecc71', 'Constant Low  ε=0.05'),
    'decay': ('#3498db', 'Decaying ε: 1.0→0.1'),
}

# ── Left: Raw rewards ────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(rewards_high,  alpha=0.35, color=colors_map['high'][0])
ax.plot(rewards_low,   alpha=0.35, color=colors_map['low'][0])
ax.plot(rewards_decay, alpha=0.35, color=colors_map['decay'][0])
ax.set_title("Raw Reward per Episode")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.legend([
    mpatches.Patch(color=colors_map['high'][0],  label=colors_map['high'][1]),
    mpatches.Patch(color=colors_map['low'][0],   label=colors_map['low'][1]),
    mpatches.Patch(color=colors_map['decay'][0], label=colors_map['decay'][1]),
], handles=[
    mpatches.Patch(color=colors_map['high'][0],  label=colors_map['high'][1]),
    mpatches.Patch(color=colors_map['low'][0],   label=colors_map['low'][1]),
    mpatches.Patch(color=colors_map['decay'][0], label=colors_map['decay'][1]),
])
ax.grid(True, alpha=0.3)
ax.axhline(y=-13, color='black', linestyle='--', linewidth=1, alpha=0.5,
           label='Optimal path reward (≈-13)')
ax.annotate('Optimal ≈ -13', xy=(N_EPISODES*0.6, -13), xytext=(N_EPISODES*0.6, -30),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=8)

# ── Right: Smoothed rewards ──────────────────────────────────────────────────
ax2 = axes[1]
s_high  = smooth(rewards_high,  WINDOW)
s_low   = smooth(rewards_low,   WINDOW)
s_decay = smooth(rewards_decay, WINDOW)
xs = range(WINDOW - 1, N_EPISODES)

ax2.plot(xs, s_high,  color=colors_map['high'][0],  linewidth=2,
         label=colors_map['high'][1])
ax2.plot(xs, s_low,   color=colors_map['low'][0],   linewidth=2,
         label=colors_map['low'][1])
ax2.plot(xs, s_decay, color=colors_map['decay'][0], linewidth=2,
         label=colors_map['decay'][1])
ax2.axhline(y=-13, color='black', linestyle='--', linewidth=1, alpha=0.6,
            label='Optimal path (≈-13)')
ax2.set_title(f"Smoothed Reward (window={WINDOW})")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Avg Total Reward")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = "/home/claude/assignment/[YourName]-[YourRoll]/reward_comparison.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Final performance stats
# ─────────────────────────────────────────────────────────────────────────────

last_n = 50
print("\n" + "="*60)
print(f"Average reward over last {last_n} episodes:")
print(f"  Constant High ε=0.50 : {np.mean(rewards_high[-last_n]):.2f}")
print(f"  Constant Low  ε=0.05 : {np.mean(rewards_low[-last_n]):.2f}")
print(f"  Decaying ε (1.0→0.1) : {np.mean(rewards_decay[-last_n]):.2f}")
print("="*60)
print("\nDone! See reward_comparison.png for the plot.")
