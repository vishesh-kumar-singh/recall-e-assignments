"""
Task 2: The Epsilon-Decay Challenge
====================================
Tabular Q-learning on CliffWalking-v0 with three exploration strategies:
  1. Constant high exploration  (epsilon = 0.5)
  2. Constant low exploration   (epsilon = 0.05)
  3. Decaying exploration       (epsilon: 1.0 → 0.01, exponential decay)

Includes a self-contained CliffWalking environment that matches the
Gymnasium CliffWalking-v0 spec, so no external RL library is needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# 1.  CliffWalking Environment (Gymnasium-spec)
# ─────────────────────────────────────────────

class CliffWalkingEnv:
    """
    4-row × 12-column grid world.
    
    Layout (row 3 = bottom):
      S  [cliff…cliff]  G
      
    - Start (S): state 36  (row 3, col 0)
    - Goal  (G): state 47  (row 3, col 11)
    - Cliff: states 37-46  (row 3, cols 1-10)
    - Actions: 0=Up, 1=Right, 2=Down, 3=Left
    - Step reward      : -1
    - Cliff reward     : -100  (agent teleported back to S)
    - Episode ends when the agent reaches G
    """

    ROWS, COLS = 4, 12
    N_STATES   = ROWS * COLS          # 48
    N_ACTIONS  = 4                    # Up Right Down Left
    START      = 36
    GOAL       = 47
    CLIFF      = set(range(37, 47))

    # Direction deltas: Up, Right, Down, Left
    _DELTA = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self):
        self.state = self.START

    def reset(self):
        self.state = self.START
        return self.state

    def step(self, action):
        r, c = divmod(self.state, self.COLS)
        dr, dc = self._DELTA[action]
        nr = max(0, min(self.ROWS - 1, r + dr))
        nc = max(0, min(self.COLS - 1, c + dc))
        next_state = nr * self.COLS + nc

        if next_state in self.CLIFF:
            reward, terminated = -100, False
            self.state = self.START
            return self.START, reward, terminated

        if next_state == self.GOAL:
            reward, terminated = -1, True
        else:
            reward, terminated = -1, False

        self.state = next_state
        return next_state, reward, terminated


# ─────────────────────────────────────────────
# 2.  Q-Learning Agent
# ─────────────────────────────────────────────

class QLearningAgent:
    """
    Tabular Q-learning with configurable epsilon strategy.

    epsilon_mode options
    --------------------
    'constant' : epsilon stays fixed at epsilon_start throughout.
    'decay'    : epsilon decays exponentially from epsilon_start
                 to epsilon_min over n_episodes.
    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.5,           # learning rate
        gamma=0.99,          # discount factor
        epsilon_start=0.5,
        epsilon_min=0.5,
        epsilon_mode="constant",
        n_episodes=500,
    ):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_mode  = epsilon_mode
        self.n_episodes    = n_episodes

        self.Q = np.zeros((n_states, n_actions))

    def get_epsilon(self, episode):
        if self.epsilon_mode == "constant":
            return self.epsilon_start
        # Exponential decay
        decay_rate = -np.log(self.epsilon_min / self.epsilon_start) / self.n_episodes
        eps = self.epsilon_start * np.exp(-decay_rate * episode)
        return max(eps, self.epsilon_min)

    def choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)          # explore
        return int(np.argmax(self.Q[state]))                  # exploit

    def update(self, state, action, reward, next_state, terminated):
        best_next = 0.0 if terminated else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])


# ─────────────────────────────────────────────
# 3.  Training Loop
# ─────────────────────────────────────────────

def train(agent, n_episodes=500, max_steps=500):
    """Train an agent and return per-episode total rewards."""
    env = CliffWalkingEnv()
    rewards_per_episode = []

    for ep in range(n_episodes):
        state   = env.reset()
        epsilon = agent.get_epsilon(ep)
        total_reward = 0

        for _ in range(max_steps):
            action = agent.choose_action(state, epsilon)
            next_state, reward, terminated = env.step(action)
            agent.update(state, action, reward, next_state, terminated)
            state        = next_state
            total_reward += reward
            if terminated:
                break

        rewards_per_episode.append(total_reward)

    return rewards_per_episode


# ─────────────────────────────────────────────
# 4.  Run all three agents
# ─────────────────────────────────────────────

N_EPISODES = 500
ALPHA      = 0.5
GAMMA      = 0.99
SEED       = 42
np.random.seed(SEED)

env_spec = dict(
    n_states  = CliffWalkingEnv.N_STATES,
    n_actions = CliffWalkingEnv.N_ACTIONS,
    alpha     = ALPHA,
    gamma     = GAMMA,
    n_episodes= N_EPISODES,
)

print("Training Agent 1: Constant High Exploration (ε = 0.50) ...")
agent_high = QLearningAgent(**env_spec, epsilon_start=0.50, epsilon_min=0.50,
                             epsilon_mode="constant")
rewards_high = train(agent_high, N_EPISODES)

print("Training Agent 2: Constant Low  Exploration (ε = 0.05) ...")
agent_low  = QLearningAgent(**env_spec, epsilon_start=0.05, epsilon_min=0.05,
                             epsilon_mode="constant")
rewards_low  = train(agent_low, N_EPISODES)

print("Training Agent 3: Decaying Exploration (ε: 1.0 → 0.01) ...")
agent_decay= QLearningAgent(**env_spec, epsilon_start=1.00, epsilon_min=0.01,
                             epsilon_mode="decay")
rewards_decay= train(agent_decay, N_EPISODES)

print("Training complete.\n")


# ─────────────────────────────────────────────
# 5.  Smoothing helper
# ─────────────────────────────────────────────

def smooth(data, window=20):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


# ─────────────────────────────────────────────
# 6.  Plot
# ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 9),
                         gridspec_kw={"height_ratios": [3, 1]})
fig.patch.set_facecolor("#0f0f1a")

ax = axes[0]
ax.set_facecolor("#0f0f1a")

episodes = np.arange(1, N_EPISODES + 1)

COLORS = {
    "high" : "#e05252",   # red
    "low"  : "#52aee0",   # blue
    "decay": "#52e08a",   # green
}

W = 25   # smoothing window

for key, rewards, label in [
    ("high",  rewards_high,  "Constant High (ε=0.50)"),
    ("low",   rewards_low,   "Constant Low  (ε=0.05)"),
    ("decay", rewards_decay, "Decaying      (ε: 1.0→0.01)"),
]:
    c = COLORS[key]
    ax.plot(episodes, rewards,        color=c, alpha=0.18, linewidth=0.8)
    ax.plot(episodes, smooth(rewards, W), color=c, alpha=0.95, linewidth=2.2,
            label=label)

ax.axhline(-13, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
ax.text(N_EPISODES * 0.98, -11, "Optimal path ≈ −13",
        color="white", alpha=0.5, ha="right", fontsize=8)

ax.set_xlim(1, N_EPISODES)
ax.set_ylim(-500, 5)
ax.set_xlabel("Episode", color="white", fontsize=11)
ax.set_ylabel("Total Reward", color="white", fontsize=11)
ax.set_title("Q-Learning on CliffWalking-v0 — Epsilon Exploration Strategies",
             color="white", fontsize=13, fontweight="bold", pad=12)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.legend(loc="lower right", facecolor="#1a1a2e", edgecolor="#555",
          labelcolor="white", fontsize=10)
ax.grid(color="#333", linewidth=0.5, alpha=0.7)

# ── Epsilon schedule subplot ──────────────────
ax2 = axes[1]
ax2.set_facecolor("#0f0f1a")

eps_high  = [0.50] * N_EPISODES
eps_low   = [0.05] * N_EPISODES
eps_decay = [agent_decay.get_epsilon(ep) for ep in range(N_EPISODES)]

ax2.plot(episodes, eps_high,  color=COLORS["high"],  linewidth=1.8)
ax2.plot(episodes, eps_low,   color=COLORS["low"],   linewidth=1.8)
ax2.plot(episodes, eps_decay, color=COLORS["decay"], linewidth=1.8)
ax2.set_xlim(1, N_EPISODES)
ax2.set_ylim(-0.05, 1.1)
ax2.set_xlabel("Episode", color="white", fontsize=10)
ax2.set_ylabel("ε value", color="white", fontsize=10)
ax2.set_title("Epsilon Schedule per Agent", color="white", fontsize=10)
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_edgecolor("#444")
ax2.grid(color="#333", linewidth=0.5, alpha=0.7)

plt.tight_layout(pad=2.0)
plt.savefig("/mnt/user-data/outputs/cliff_walking_results.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Plot saved → cliff_walking_results.png")
plt.close()


# ─────────────────────────────────────────────
# 7.  Summary statistics
# ─────────────────────────────────────────────

def final_avg(rewards, last_n=50):
    return np.mean(rewards[-last_n:])

def first_good_episode(rewards, threshold=-20):
    """First episode where reward exceeded threshold (rolling window 10)."""
    for i in range(9, len(rewards)):
        if np.mean(rewards[i-9:i+1]) >= threshold:
            return i + 1
    return None

print("=" * 52)
print(f"{'Agent':<30} {'Final avg (last 50)':<22} {'First safe ep':>13}")
print("-" * 52)
for name, rewards in [
    ("Constant High (ε=0.50)", rewards_high),
    ("Constant Low  (ε=0.05)", rewards_low),
    ("Decaying (ε: 1.0→0.01)", rewards_decay),
]:
    fa  = final_avg(rewards)
    fge = first_good_episode(rewards)
    fge_str = str(fge) if fge else "never"
    print(f"{name:<30} {fa:<22.2f} {fge_str:>13}")
print("=" * 52)