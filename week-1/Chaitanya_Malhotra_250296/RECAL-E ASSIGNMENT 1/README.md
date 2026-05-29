# Cliff Walking Q-Learning Exploration Analysis

This repository contains a Python implementation of the **Q-learning algorithm** applied to the standard **Cliff Walking** gridworld environment from OpenAI Gymnasium. 

The project evaluates how different $\epsilon$-greedy exploration policies impact an RL agent's learning rate and overall cumulative rewards over a series of training episodes.

---

## 🎮 The Environment: Cliff Walking (`v1`)

The environment consists of a $4 \times 12$ grid. The agent starts at the bottom-left corner and aims to reach the bottom-right destination. 

* **The Cliff:** The tiles between the start and destination represent a deadly cliff.
* **Rewards:** * Each safe step yields a reward of `-1`.
  * Falling into the cliff yields a penalty of `-100` and sends the agent right back to the starting position.

---

## 📈 Exploration Strategies Tested

The script executes and compares three distinct parameter configurations to visualize how balancing **exploration vs. exploitation** shifts training stability:

1. **`x1` - Linear Epsilon Decay (Balanced):** Starts with maximum exploration ($\epsilon = 1.0$) and gradually reduces it by $0.01$ each episode. This allows the agent to thoroughly discover the map early on and transition to optimal play later. The probability of taking a random action is $\epsilon$ which decays linearly thus reducing the chances of taking actions randomly and using Q-table frequently to determine actions giving maximum Q-value. Thus it explores first thouroughly and exploits later
2. **`x2` - Constant High Exploration (No Decay):** Keeps $\epsilon = 1.0$ permanently. The agent takes completely random actions throughout the entire training cycle, demonstrating baseline search behaviors without maximizing policies. Thus there is no such exploitation as it never uses Q-Table.
3. **`x3` - Pure Exploitation (No Exploration):** Starts with a static, near-zero exploration rate ($\epsilon = 0.05$). The agent relies entirely on its zero-initialized Q-table, resulting in poor trajectory discoveries.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with the required libraries. You can install the specialized Gymnasium package using:

```bash
pip install "gymnasium[classic-control]" numpy matplotlib