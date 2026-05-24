# Task 2 — The Epsilon-Decay Challenge

Tabular Q-learning on the **CliffWalking-v0** environment with three distinct
epsilon (exploration) strategies, analysed side-by-side.

---

## Table of Contents
1. [Environment](#environment)
2. [Project Structure](#project-structure)
3. [Setup & Run](#setup--run)
4. [Agents & Exploration Strategies](#agents--exploration-strategies)
5. [Algorithm](#algorithm)
6. [Results & Analysis](#results--analysis)
7. [Conclusions](#conclusions)

---

## Environment

**CliffWalking-v0** is a classic grid-world from Sutton & Barto (Example 6.6).

```
 ┌────────────────────────────────────┐
 │  .   .   .   .   .   .   .   .   .   .   .   .  │  row 0
 │  .   .   .   .   .   .   .   .   .   .   .   .  │  row 1
 │  .   .   .   .   .   .   .   .   .   .   .   .  │  row 2
 │  S  [C  C  C  C  C  C  C  C  C  C] G  │  row 3
 └────────────────────────────────────┘
       col 0                        col 11
```

| Symbol | Meaning |
|--------|---------|
| `S`    | Start (state 36) |
| `G`    | Goal  (state 47) |
| `C`    | Cliff (states 37–46) |

**Reward structure**

| Event | Reward |
|-------|--------|
| Regular step | −1 |
| Walking into the cliff | −100 (reset to S, episode continues) |
| Reaching the goal | −1 (episode ends) |

**Optimal path reward**: −13 (walk along the top, 13 steps).  
**Safe-but-suboptimal path reward**: −17 (hug the top row, avoid cliff).

The environment is **fully implemented from scratch** in the script — no
external Gymnasium installation is required. It matches the Gymnasium
`CliffWalking-v0` specification exactly.

---

## Project Structure

```
.
├── q_learning_cliff.py          # Main script (env + agents + training + plot)
├── cliff_walking_results.png    # Output chart (generated on run)
└── README.md                    # This file
```

---

## Setup & Run

### Requirements
- Python ≥ 3.8
- `numpy`
- `matplotlib`

```bash
pip install numpy matplotlib
```

> **No Gymnasium needed.** The CliffWalking environment is self-contained.

### Run

```bash
python q_learning_cliff.py
```

This will:
1. Train all three agents for 500 episodes each.
2. Print a summary table to the console.
3. Save `cliff_walking_results.png` in the current directory.

---

## Agents & Exploration Strategies

Three agents are trained, identical in every way **except** their epsilon strategy.

| Agent | Strategy | Epsilon |
|-------|----------|---------|
| **Agent 1** | Constant High | ε = 0.50 (fixed) |
| **Agent 2** | Constant Low  | ε = 0.05 (fixed) |
| **Agent 3** | Decaying      | ε: 1.0 → 0.01 (exponential) |

### Epsilon Decay Formula (Agent 3)

```
decay_rate = -ln(ε_min / ε_start) / N_episodes
ε(episode) = ε_start × exp(-decay_rate × episode)
```

With `ε_start = 1.0`, `ε_min = 0.01`, `N = 500`:

- Episode   0 → ε ≈ 1.000  (full exploration)
- Episode 250 → ε ≈ 0.100
- Episode 500 → ε ≈ 0.010  (near-greedy)

---

## Algorithm

### Tabular Q-Learning (off-policy TD control)

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]
```

| Hyper-parameter | Value |
|-----------------|-------|
| Learning rate α | 0.50  |
| Discount γ      | 0.99  |
| Episodes        | 500   |
| Max steps/ep    | 500   |
| Q init          | 0     |

**Action selection**: ε-greedy — with probability ε choose a random action,
otherwise choose `argmax Q(s, ·)`.

---

## Results & Analysis

### Console Summary (typical run)

```
====================================================
Agent                          Final avg (last 50)    First safe ep
----------------------------------------------------
Constant High (ε=0.50)         -270.80                        never
Constant Low  (ε=0.05)          -21.64                           73
Decaying (ε: 1.0→0.01)          -15.56                          245
====================================================
```

### Chart

The output chart (`cliff_walking_results.png`) contains two panels:

- **Top**: Sum of rewards per episode (raw + smoothed rolling average, window=25).
- **Bottom**: Epsilon schedule for each agent across all 500 episodes.

---

## Conclusions

### Which agent learns a safe path the fastest?

**Agent 2 (Constant Low, ε = 0.05)** finds a safe path earliest — typically
around episode 70–80. Because it exploits from very early on, it quickly locks
onto a reliable (though not perfectly optimal) route that avoids the cliff.

### Which agent ultimately finds the most optimal path?

**Agent 3 (Decaying ε)** achieves the highest long-run performance (closest
to −13). By starting with full exploration it maps out the entire state space,
and as epsilon anneals to near-zero it commits to the best discovered policy.

### Why is there a difference?

The difference comes down to the **exploration–exploitation trade-off**:

| Agent | Behaviour | Consequence |
|-------|-----------|-------------|
| **High ε = 0.50** | Always 50 % random actions, forever | Keeps stumbling into the cliff even after learning; never converges to a good policy. Final average stays deeply negative. |
| **Low ε = 0.05** | Almost purely greedy from episode 1 | Quickly settles on a *safe* but *conservative* path (hugging the top row) because it never explored enough to learn the true value of riskier, shorter routes. |
| **Decaying ε** | Starts exploratory, gradually becomes greedy | Gets the best of both worlds: sufficient early exploration to discover optimal values, then exploitation to consistently execute the optimal policy. |

This exactly replicates the classic finding from Sutton & Barto: Q-learning
with ε-greedy converges to the **optimal policy** only when ε → 0 over time,
but a fixed small ε can still learn a *near*-optimal policy faster in
wall-clock episodes due to less wasted exploration.

---

*Implementation by: Q-Learning Epsilon-Decay Challenge, Task 2*
