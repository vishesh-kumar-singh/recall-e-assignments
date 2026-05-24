"""
task2_qlearning.py
------------------
Tabular Q-learning on CliffWalking-v1 with three exploration strategies:
  1. Constant high epsilon  (eps = 0.5)
  2. Constant low epsilon   (eps = 0.05)
  3. Decaying epsilon       (1.0 → 0.01)

Saves:
  - reward_curves.png  : smoothed reward curves for all three agents
  - q_tables.npy       : Q-tables for all three agents (dict)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gymnasium as gym

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED        = 42
N_EPISODES  = 1000
MAX_STEPS   = 200      # safety cap per episode
ALPHA       = 0.1      # learning rate
GAMMA       = 0.99     # discount factor
SMOOTH_WIN  = 30       # window for rolling average in plot

rng = np.random.default_rng(SEED)


# ── Q-learning agent ──────────────────────────────────────────────────────────
def run_agent(eps_fn, n_episodes=N_EPISODES, seed=SEED):
    """
    Train a tabular Q-learning agent.

    Parameters
    ----------
    eps_fn : callable  int -> float
        Returns epsilon given the episode number (0-indexed).

    Returns
    -------
    rewards : np.ndarray  shape (n_episodes,)
        Sum of rewards per episode.
    Q : np.ndarray  shape (n_states, n_actions)
        Learned Q-table.
    """
    env = gym.make("CliffWalking-v1")
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0
        eps = eps_fn(ep)

        for _ in range(MAX_STEPS):
            # ε-greedy action selection
            if rng.random() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-update
            best_next = np.max(Q[next_state])
            Q[state, action] += ALPHA * (
                reward + GAMMA * best_next - Q[state, action]
            )

            state      = next_state
            ep_reward += reward

            if done:
                break

        rewards[ep] = ep_reward

    env.close()
    return rewards, Q


# ── Epsilon schedules ─────────────────────────────────────────────────────────
def eps_high(_ep):       return 0.5
def eps_low(_ep):        return 0.05
def eps_decay(ep):
    start, end = 1.0, 0.01
    return max(end, start - (start - end) * ep / N_EPISODES)


# ── Train all three agents ────────────────────────────────────────────────────
print("Training Agent 1: Constant high epsilon (ε = 0.5) …")
r_high,  Q_high  = run_agent(eps_high)

print("Training Agent 2: Constant low epsilon  (ε = 0.05) …")
r_low,   Q_low   = run_agent(eps_low)

print("Training Agent 3: Decaying epsilon (1.0 → 0.01) …")
r_decay, Q_decay = run_agent(eps_decay)

print("Training complete.")


# ── Save Q-tables ─────────────────────────────────────────────────────────────
np.save("q_tables.npy", {"high": Q_high, "low": Q_low, "decay": Q_decay})
print("Q-tables saved to q_tables.npy")


# ── Smoothing helper ──────────────────────────────────────────────────────────
def smooth(x, w=SMOOTH_WIN):
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")

episodes = np.arange(1, N_EPISODES + 1)
sm_ep    = episodes[SMOOTH_WIN - 1:]   # x-axis for smoothed curves

palette = {
    "high":  "#ff6b6b",
    "low":   "#4ecdc4",
    "decay": "#ffe66d",
}

# raw (faint)
for r, key in [(r_high, "high"), (r_low, "low"), (r_decay, "decay")]:
    ax.plot(episodes, r, color=palette[key], alpha=0.12, linewidth=0.7)

# smoothed
ax.plot(sm_ep, smooth(r_high),  color=palette["high"],  linewidth=2.2,
        label=f"Constant High ε = 0.5")
ax.plot(sm_ep, smooth(r_low),   color=palette["low"],   linewidth=2.2,
        label=f"Constant Low ε = 0.05")
ax.plot(sm_ep, smooth(r_decay), color=palette["decay"], linewidth=2.5,
        label=f"Decaying ε (1.0 → 0.01)", zorder=5)

# Optimal reward reference
ax.axhline(-13, color="white", linestyle="--", linewidth=1, alpha=0.4,
           label="Optimal path reward (−13)")

ax.set_xlabel("Episode", color="#ccccdd", fontsize=11)
ax.set_ylabel("Sum of Rewards", color="#ccccdd", fontsize=11)
ax.set_title("CliffWalking-v1: Exploration Strategy Comparison\n"
             "(Tabular Q-learning, α=0.1, γ=0.99, smoothed over 30 episodes)",
             color="white", fontsize=13, pad=14)

ax.tick_params(colors="#aaaacc")
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")

ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.grid(True, color="#222244", linewidth=0.6)
legend = ax.legend(fontsize=10, framealpha=0.3, facecolor="#1a1a2e",
                   edgecolor="#555577", labelcolor="white")

plt.tight_layout()
plt.savefig("reward_curves.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Plot saved to reward_curves.png")


# ── Terminal summary ──────────────────────────────────────────────────────────
last = 100   # look at last 100 episodes
print("\n── Final 100-episode average rewards ──")
print(f"  High ε  (0.50) : {r_high[-last:].mean():.1f}")
print(f"  Low  ε  (0.05) : {r_low[-last:].mean():.1f}")
print(f"  Decay ε (→0.01): {r_decay[-last:].mean():.1f}")
