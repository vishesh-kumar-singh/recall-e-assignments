import numpy as np
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except ImportError:
    import gym

N_EPISODES  = 500
MAX_STEPS   = 200
ALPHA       = 0.5
GAMMA       = 0.99
SEED        = 42

EPS_START   = 1.0
EPS_END     = 0.01
DECAY_MODE  = "exponential"
EPS_DECAY_RATE = np.exp(np.log(EPS_END / EPS_START) / (N_EPISODES - 1))


def get_epsilon_constant_high(_episode):
    return 0.50

def get_epsilon_constant_low(_episode):
    return 0.05

def get_epsilon_decay(episode):
    if DECAY_MODE == "linear":
        return max(EPS_END, EPS_START - (EPS_START - EPS_END) * episode / (N_EPISODES - 1))
    return max(EPS_END, EPS_START * (EPS_DECAY_RATE ** episode))


def run_qlearning(eps_fn, seed=SEED):
    env = gym.make("CliffWalking-v1")
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []
    rng = np.random.default_rng(seed)

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0
        epsilon = eps_fn(ep)

        for _ in range(MAX_STEPS):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[obs]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next = np.max(Q[next_obs])
            Q[obs, action] += ALPHA * (reward + GAMMA * best_next - Q[obs, action])

            obs = next_obs
            total_reward += reward

            if done:
                break

        rewards_per_episode.append(total_reward)

    env.close()
    return rewards_per_episode, Q


print("Training Agent 1 – Constant High Exploration (eps=0.50)...")
rewards_high,  Q_high  = run_qlearning(get_epsilon_constant_high)

print("Training Agent 2 – Constant Low  Exploration (eps=0.05)...")
rewards_low,   Q_low   = run_qlearning(get_epsilon_constant_low)

print(f"Training Agent 3 – Decaying Exploration ({DECAY_MODE})...")
rewards_decay, Q_decay = run_qlearning(get_epsilon_decay)

print("Training complete.\n")


def smooth(data, window=20):
    return np.convolve(data, np.ones(window) / window, mode='valid')


WINDOW = 20
episodes_smooth = np.arange(WINDOW - 1, N_EPISODES)

fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(
    "Q-learning on CliffWalking-v1\nEpsilon Exploration Strategy Comparison",
    fontsize=14, fontweight='bold', y=0.98
)

ax = axes[0]
colors_map = {'high': '#e74c3c', 'low': '#2980b9', 'decay': '#27ae60'}

for rewards, key, label in [
    (rewards_high,  'high',  'Constant High (ε=0.50)'),
    (rewards_low,   'low',   'Constant Low  (ε=0.05)'),
    (rewards_decay, 'decay', f'Decaying  (ε: 1.0→0.01, {DECAY_MODE})'),
]:
    ax.plot(rewards, alpha=0.15, color=colors_map[key])
    ax.plot(episodes_smooth, smooth(rewards, WINDOW),
            color=colors_map[key], linewidth=2.2, label=label)

ax.axhline(-13, color='grey', linestyle='--', linewidth=1, label='Optimal path reward (−13)')
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Sum of Rewards per Episode", fontsize=11)
ax.set_title("Learning Curves (bold = 20-episode moving average)", fontsize=11)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, N_EPISODES)

ax2 = axes[1]
ep_range = np.arange(N_EPISODES)
ax2.plot(ep_range, [get_epsilon_constant_high(e) for e in ep_range], color=colors_map['high'],  linewidth=1.8, label='High')
ax2.plot(ep_range, [get_epsilon_constant_low(e)  for e in ep_range], color=colors_map['low'],   linewidth=1.8, label='Low')
ax2.plot(ep_range, [get_epsilon_decay(e)          for e in ep_range], color=colors_map['decay'], linewidth=1.8, label='Decay')
ax2.set_xlabel("Episode", fontsize=11)
ax2.set_ylabel("ε (Epsilon)", fontsize=11)
ax2.set_title("Epsilon Schedules", fontsize=11)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, N_EPISODES)
ax2.set_ylim(-0.05, 1.1)

plt.tight_layout()
plt.savefig("epsilon_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved to epsilon_comparison.png")


LAST_N = 50
print("\n===== Analysis =====")
for rewards, label in [
    (rewards_high,  "Constant High (ε=0.50)"),
    (rewards_low,   "Constant Low  (ε=0.05)"),
    (rewards_decay, f"Decaying (ε 1.0→0.01)"),
]:
    last_mean = np.mean(rewards[-LAST_N:])
    rolling = smooth(rewards, WINDOW)
    safe_eps = next((i for i, v in enumerate(rolling) if v > -20), None)
    safe_eps = safe_eps + WINDOW - 1 if safe_eps is not None else "never"
    print(f"  {label}")
    print(f"    Mean reward (last {LAST_N} eps): {last_mean:.2f}")
    print(f"    First episode with rolling avg > -20: {safe_eps}")
    print()
