# Week 1 Assignment


# Repository Structure

```text
week-1/
└── bhavna-250275/
    ├── task-1.pdf
    ├── README.md
    └── task-2/
        ├── q_learning_cliffwalking.py
        └── rewards_plot.png
```

---

# Task 1: Formulating the World as an MDP

## Scenario: Smart Traffic Signal Control

A smart traffic signal system controls traffic flow at a busy road intersection. The objective is to reduce congestion while maintaining smooth traffic movement and pedestrian safety.

The traffic controller can:
- Keep the signal green
- Switch traffic direction
- Activate pedestrian crossing mode

The goal is to maximize smooth traffic flow and minimize congestion.

---

# State Space (S)

| State | Meaning |
|---|---|
| Light | Low traffic |
| Moderate | Normal traffic |
| Heavy | High traffic congestion |

\[
S = \{Light,\ Moderate,\ Heavy\}
\]

---

# Action Space (A)

| Action | Meaning |
|---|---|
| KeepGreen | Continue current green signal |
| SwitchSignal | Change traffic direction |
| PedestrianMode | Enable pedestrian crossing |

\[
A = \{KeepGreen,\ SwitchSignal,\ PedestrianMode\}
\]

---

# Reward Function (R)

| Situation | Reward |
|---|---|
| Smooth traffic flow | +10 |
| Reduced congestion | +6 |
| Pedestrian crossing success | +4 |
| Heavy congestion | -8 |

---

# Task 2: Q-Learning on CliffWalking-v0

## Requirements

Install dependencies using:

```bash
pip install gymnasium matplotlib numpy
```

---

# Run the Program

From inside the `task-2` folder:

```bash
python q_learning_cliffwalking.py
```

OR from the main assignment folder:

```bash
python task-2/q_learning_cliffwalking.py
```

---

# Output

The program generates:
- rewards_plot.png

---

# Analysis

## Constant High Exploration (ε = 0.5)

- Explores heavily
- Learns slowly
- Frequently falls into the cliff

---

## Constant Low Exploration (ε = 0.1)

- Learns safer paths quickly
- Less random exploration
- May converge to suboptimal policies

---

## Decaying Exploration

- Starts with high exploration
- Gradually reduces exploration
- Learns efficiently
- Achieves near-optimal performance

---

# Conclusion

- The low exploration agent learns the safe path fastest.
- The decaying exploration agent performs best overall because it balances:
  - exploration in early episodes
  - exploitation in later episodes