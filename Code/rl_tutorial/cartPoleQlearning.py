import sys
sys.path.append('/home/eric/gym')
import gym
import numpy as np
import math
import matplotlib.pyplot as plt

env=gym.make('CartPole-v0')
n_actions=env.action_space.n
n_states=env.observation_space.shape[0]
print('Action space size: ',n_actions)
print('State space size: ',n_actions)

print('states high value:')
print(env.observation_space.high[0])
print(env.observation_space.high[1])
print(env.observation_space.high[2])
print(env.observation_space.high[3])

print('states low value:')
print(env.observation_space.low[0])
print(env.observation_space.low[1])
print(env.observation_space.low[2])
print(env.observation_space.low[3])



