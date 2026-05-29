import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Hyperparameters ---
EPISODES = 500
ALPHA = 0.1       # Learning rate
GAMMA = 0.99      # Discount factor

def train_q_learning(agent_type):
    """
    Trains a Q-learning agent on CliffWalking-v1.
    agent_type can be: 'high_eps', 'low_eps', or 'decay_eps'
    """
    # FIX 1: Corrected environment ID to v0
    env = gym.make('CliffWalking-v1')
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    Q = np.zeros((num_states, num_actions))
    episode_rewards = []
    
    # FIX 2: Corrected initial epsilon values per assignment guidelines
    if agent_type == 'high_eps':
        epsilon = 0.5
    elif agent_type == 'low_eps':
        epsilon = 0.05
    else:  # decay_eps
        epsilon = 1.0

    for episode in range(EPISODES):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        # FIX 2 (Cont.): Linear decay setup from 1.0 down to 0.01 over 450 episodes
        if agent_type == 'decay_eps':
            epsilon = max(0.01, 1.0 - (episode / 450.0))

        while not done:
            # 1. Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample() # Explore
            else:
                action = np.argmax(Q[state])       # Exploit
                
            # 2. Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 3. Q-Table Update Rule
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + GAMMA * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += ALPHA * td_error
            
            state = next_state
            total_reward += reward
            
        episode_rewards.append(total_reward)
        
    env.close()
    return episode_rewards

# --- Helper function to smooth out the noisy reward curves ---
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# --- Train the Three Separate Agents ---
print("Training High Exploration Agent (eps=0.5)...")
high_eps_rewards = train_q_learning('high_eps')

print("Training Low Exploration Agent (eps=0.05)...")
low_eps_rewards = train_q_learning('low_eps')

print("Training Decaying Exploration Agent (1.0 -> 0.01)...")
decay_eps_rewards = train_q_learning('decay_eps')

# --- Plotting the Results ---
plt.figure(figsize=(10, 6))

# FIX 3: Plotting smoothed curves so your assignment plot is crystal clear
plt.plot(moving_average(high_eps_rewards), label='Constant High (eps=0.5)', alpha=0.7)
plt.plot(moving_average(low_eps_rewards), label='Constant Low (eps=0.05)', alpha=0.7)
plt.plot(moving_average(decay_eps_rewards), label='Decaying eps (1.0 -> 0.01)', alpha=0.9, linewidth=2)

plt.xlabel('Episodes (Smoothed)')
plt.ylabel('Sum of Rewards')
plt.title('Q-Learning Exploration Strategy Comparison (CliffWalking-v0)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Save the plot automatically for your README markdown
plt.savefig('learning_plot.png', dpi=300)
print("Training complete! 'learning_plot.png' has been saved.")
plt.show()