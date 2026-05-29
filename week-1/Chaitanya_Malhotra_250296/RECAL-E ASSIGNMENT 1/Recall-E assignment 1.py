# Run `pip install "gymnasium[classic-control]"` for this example.
import time

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# Create our training environment - a cart with a pole that needs balancing
def run(ep,epsilon,ep_decay_rate):
    i = 0
    
    n_rows = 4
    n_cols = 12
    n_actions = 4
    rnd = np.random.default_rng()
    lr = 0.1
    discount = 0.95
    value = np.zeros((n_rows* n_cols , n_actions))
    reward_per_episode = np.zeros(ep)
    
    env = gym.make("CliffWalking-v1", render_mode=None)
    while(i<ep):
        print("Episode: ", i+1)
        total_reward = 0
        truncated = False
        terminated = False
       
        
        
        observation, info = env.reset()
        while not truncated and not terminated:
            if(rnd.random() < epsilon):
                action = env.action_space.sample()  # Your agent here (this takes random actions)
            else:
                action = np.argmax(value[observation])
            new_observation, reward, terminated, truncated, info = env.step(action)
            #print(observation, total_reward, terminated, truncated, info, action)
            value[observation][action] += lr* (reward + discount * np.max(value[new_observation]) - value[observation][action])
            total_reward += reward
            observation = new_observation
    
        epsilon = max(epsilon-ep_decay_rate,0)   # LINEAR DECAY
        #epsilon = ep_decay_rate*epsilon           # EXPONENTIAL DECAY
        reward_per_episode[i] = total_reward
        i+=1
        print("Total Reward: ", total_reward)
    env.close()
    """render_env = gym.make("CliffWalking-v1", render_mode="human")
    truncated = False
    terminated = False
    observation, info = render_env.reset()
    while not truncated and not terminated:
        action = np.argmax(value[observation])
        new_observation, reward, terminated, truncated, info = render_env.step(action)
        observation = new_observation
        time.sleep(0.5)
    render_env.close()


    plt.plot(reward_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Sum of Rewards")
    plt.title("Training Progress")
    plt.show()"""

    return reward_per_episode

x1 = run(150, 1, 0.01)
x2 = run(150, 1, 0)
x3 = run(150, 0.05, 0)

plt.plot(x1, label="Linear Decay")
plt.plot(x2, label="No Decay")
plt.plot(x3, label="No Exploration")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Sum of Rewards")   
plt.savefig("exploration_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
