#  Q-Learning on CliffWalking-v1 (Epsilon Decay Challenge)

This project implements **Tabular Q-Learning** on the classic reinforcement learning environment **CliffWalking-v1** from Gymnasium. The goal is to compare different exploration strategies and analyze their effect on learning performance.

---

#  Problem Statement

We train three Q-learning agents with different exploration strategies and compare their performance:

1.  Constant High Exploration (ε = 0.5)
2.  Constant Low Exploration (ε = 0.05)
3.  Decaying Exploration (ε: 1.0 → 0.01)

We analyze:

* Reward per episode
* Stability of learning
* Speed of convergence
* Quality of learned policy

---

#  Environment Details

We use the **Cliff Walking environment** from entity["software","Gymnasium","Python reinforcement learning library"].

### Key Features:

* Grid world environment
* Start → Goal navigation
* Cliff region with heavy penalty (-100)
* Episode ends when agent reaches goal or falls into cliff

---

#  Q-Learning Algorithm

We use **Tabular Q-Learning**, a model-free reinforcement learning method.

### Update Rule:

Q-learning updates values using:

Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]

---

##  Parameters Used

| Parameter | Value | Meaning                |
| --------- | ----- | ---------------------- |
| α (Alpha) | 0.1   | Learning rate          |
| γ (Gamma) | 0.99  | Discount factor        |
| Episodes  | 500   | Training iterations    |
| Max Steps | 100   | Step limit per episode |

---

#  Exploration Strategies

## 1.  Constant High Exploration (ε = 0.5)

* Agent chooses random action 50% of the time
* Strong exploration throughout training
* Slower convergence

### Behavior:

* Learns environment slowly
* Frequently falls into cliff
* High variance in rewards

---

## 2.  Constant Low Exploration (ε = 0.05)

* Mostly exploits known knowledge
* Very little exploration

### Behavior:

* Learns quickly at start
* Can get stuck in suboptimal safe path
* Poor exploration of better routes

---

## 3.  Decaying Exploration (1.0 → 0.01)

* Starts with full exploration
* Gradually shifts to exploitation

### Behavior:

* Explores environment thoroughly early
* Learns safe path structure
* Eventually converges to optimal policy

---

# 📈 Training Output

For each episode, we record:

* Total reward
* Episode completion status

These are plotted as:

📊 **Reward vs Episode graph for all 3 agents**

> Insert plot here:

```
[Q-learning on CliffWalking.png]
```

---

#  Expected Graph Interpretation

###  High Exploration

* Noisy reward curve
* Slow improvement
* Never fully stabilizes

###  Low Exploration

* Fast early improvement
* Plateaus early
* Suboptimal convergence

###  Decaying Exploration

* Slow start
* Rapid improvement after learning phase
* Highest final performance

---

#  Key Observations

### 1. Fastest Learning Agent

 Constant Low Exploration (ε = 0.05)

* Exploits early knowledge quickly
* But may not explore enough to find best path

---

### 2. Most Stable & Optimal Agent

 Decaying Exploration

* Best balance between exploration and exploitation
* Learns environment first, optimizes later
* Achieves best long-term performance

---

### 3. Why Difference Occurs?

| Strategy | Problem                  |
| -------- | ------------------------ |
| High ε   | Too much randomness      |
| Low ε    | Insufficient exploration |
| Decay ε  | Balanced learning        |

---

#  Final Conclusion

The experiment shows that **decaying epsilon is the most effective strategy** because it:

* Encourages early exploration
* Reduces randomness over time
* Allows stable convergence to optimal policy

This matches standard reinforcement learning theory where exploration is needed early and exploitation later.

---

#  How to Run

```bash
python The Epsilon Decay Challenge.py
```

---

#  Project Structure

```
project/
│── The Epsilon Decay Challenge.py
│── README.md
│── Q-learning on CliffWalking.png
```

---

#  Author Notes

This project demonstrates core RL concepts:

* Q-learning
* Epsilon-greedy policy
* Exploration vs Exploitation tradeoff
* Reward shaping and convergence behavior
