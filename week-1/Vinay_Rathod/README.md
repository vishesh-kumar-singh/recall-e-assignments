Analysis

The performance of the three Q-learning agents differs because of how each balances the exploration–exploitation tradeoff.

1. Constant High Exploration (ϵ=0.5)

The agent with constant high exploration performs the worst overall. Since the exploration rate remains very high throughout training, the agent continues taking many random actions even after learning useful Q-values. As a result, it frequently falls into the cliff region, causing highly negative rewards and unstable learning behavior.

Although this strategy explores the environment extensively, it is unable to consistently exploit the optimal policy. This is reflected in the graph by the noisy reward curve and lower average rewards.

2. Constant Low Exploration (ϵ=0.05)

The constant low exploration agent performs significantly better than the high exploration agent. Because the exploration rate is small, the agent mostly exploits previously learned actions and avoids unnecessary risky moves.

This allows the rewards to stabilize relatively quickly. However, due to limited exploration from the beginning, the agent may converge to a suboptimal safe path without fully discovering better routes near the cliff.

Thus, the agent achieves stable but slightly less optimal performance compared to the decaying exploration strategy.

3. Decaying Exploration

The decaying exploration agent achieves the best overall performance. Initially, the agent explores heavily, which helps it efficiently discover the environment, including dangerous cliff states and better paths to the goal.

As training progresses, the exploration rate gradually decreases, allowing the agent to increasingly exploit the best learned policy. This combination of early exploration and later exploitation enables faster learning and better convergence.

From the graph, the decaying exploration strategy:

learns a safe path the fastest,
achieves the highest rewards,
and maintains stable performance over time.
Conclusion

Among the three strategies, the decaying exploration approach performs the best overall because it effectively balances exploration and exploitation. High exploration prevents stable convergence, while very low exploration can limit discovery of optimal policies. Decaying exploration overcomes both issues by exploring early in training and exploiting learned knowledge later.


<img src="https://github.com/user-attachments/assets/9d7a3486-efb7-4f3a-b648-1e4588cad6d6" width="700"/>
