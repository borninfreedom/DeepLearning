import sys
sys.path.append('/home/eric/gym')
import numpy as np
import matplotlib.pyplot as plt
import gym
import random

env=gym.make("Taxi-v2")
action_size=env.action_space.n
state_size=env.observation_space.n
print('Action space size: ',action_size)
print('State space size: ',state_size)

Q=np.zeros((state_size,action_size))

n_episodes=2000
n_steps=100
alpha=0.7   #learning rate
gamma=0.618 #discounting rate

epsilon=1   #exploration rate
max_epsilon=1
min_epsilon=0.01
decay_rate=0.01 #exponential decay rate for exploration prob

rewards=[]
for episode in range(n_episodes):
    state=env.reset()
    episode_rewards=0

    for t in range(n_steps):
        action=np.argmax(Q[state,:])
        new_state,reward,done,info=env.step(action)
        Q[state,action]=Q[state,action]+reward+np.max(Q[new_state,:])
        episode_rewards+=reward
        state=new_state
        if done==True:
            print('Cumulative reward for episode {}:{}'.format(episode,episode_rewards))
            break
    rewards.append(episode_rewards)
print('Traning score over time:'+str(sum(rewards)/n_episodes))

x=range(n_episodes)
plt.plot(x,rewards)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.savefig('plots/Q_learning_simple_update.png',dpi=300)
plt.show()