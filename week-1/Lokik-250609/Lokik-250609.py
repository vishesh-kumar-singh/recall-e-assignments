import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
import matplotlib.pyplot as plt
#  Initialize the Environment this initializing was giving a depreciation error but I looked it up on an AI the way to export the environment has changed but the rest is the same
env = CliffWalkingEnv()
#  Hyperparameters
alpha = 0.1         
gamma = 0.99        
epsilon1 = 0.05      

epsilon2 = 0.5

epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.995

episodes = 500     

# Initialize Q-Table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table1 = np.zeros((num_states, num_actions))
q_table2 = np.zeros((num_states, num_actions))
q_table3 = np.zeros((num_states, num_actions))
reward_ateach_episode1 = np.zeros(500, dtype=int) 
reward_ateach_episode2 = np.zeros(500, dtype=int)
reward_ateach_episode3 = np.zeros(500, dtype=int)
#for epsilon 1
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample() # Explore
    else:
        return np.argmax(q_table1[state, :]) # Exploit

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = choose_action(state, epsilon1)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        
        # Q-Learning update rule
        best_next_action = np.argmax(q_table1[next_state, :])
        td_target = reward + gamma * q_table1[next_state, best_next_action]
        td_error = td_target - q_table1[state, action]
        q_table1[state, action] += alpha * td_error
        state = next_state
    reward_ateach_episode1[episode] = total_reward

#for epsilon 2
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample() # Explore
    else:
        return np.argmax(q_table2[state, :]) # Exploit

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = choose_action(state, epsilon2)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        
        # Q-Learning update rule
        best_next_action = np.argmax(q_table2[next_state, :])
        td_target = reward + gamma * q_table2[next_state, best_next_action]
        td_error = td_target - q_table2[state, action]
        q_table2[state, action] += alpha * td_error
        state = next_state
    reward_ateach_episode2[episode] = total_reward
#for epsilon 3
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample() # Explore
    else:
        return np.argmax(q_table3[state, :]) # Exploit

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        
        # Q-Learning update rule
        best_next_action = np.argmax(q_table3[next_state, :])
        td_target = reward + gamma * q_table3[next_state, best_next_action]
        td_error = td_target - q_table3[state, action]
        q_table3[state, action] += alpha * td_error
        state = next_state
    reward_ateach_episode3[episode] = total_reward
    epsilon = max(
        min_epsilon,
        epsilon * decay_rate
    )
plt.scatter(range(len(reward_ateach_episode1)), reward_ateach_episode1, marker='o', linestyle='-', color='b', label='Reward per Episode1')
plt.scatter(range(len(reward_ateach_episode2)), reward_ateach_episode2, marker='o', linestyle='-', color='r', label='Reward per Episode2')
plt.scatter(range(len(reward_ateach_episode3)), reward_ateach_episode3, marker='o', linestyle='-', color='g', label='Reward per Episode3')

# 3. Add labels and a grid so it's easy to read
plt.title('Reward per Episode in Cliff Walking Environment')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.legend()

# 4. Show the plot
plt.show()
env.close()