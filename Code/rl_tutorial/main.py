import sys
sys.path.append('/home/eric/gym')
import gym
import matplotlib.pyplot as plt

env=gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

def policy():
    """return a random action: either 0(left) or 1(right)"""
    action=env.action_space.sample()
    return action

def policy_hard_code_1(t):
    action=0
    if t<20:
        action=0
    elif t>=20:
        action=1
    return action

nb_episodes=50
nb_timesteps=100
x_list=[]
y_list=[]

for episode in range(nb_episodes):
    state=env.reset()
    rewards=[]
    x_list.append(episode+1)
    for t in range(nb_timesteps):
        env.render()
        state,reward,done,info=env.step(policy_hard_code_1(t))
        rewards.append(reward)
        if done:
            cumulative_reward=sum(rewards)
            y_list.append(cumulative_reward)
            print("episode {} finished after {} timesteps.Total reward:{}".format(episode,t+1,cumulative_reward))
            break

plt.figure()
ax=plt.gca()
ax.set_xlabel('episode')
ax.set_ylabel('rewards')
#ax.scatter(x_list,y_list,c='r',s=20,alpha=0.5)
ax.plot(x_list,y_list,c='r',linewidth=1,alpha=0.5)
plt.show()
env.close()
