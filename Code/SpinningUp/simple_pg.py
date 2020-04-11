import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import sys
sys.path.append('/home/eric/gym')
import gym
from gym.spaces import Discrete,Box

def mlp(sizes,activation=nn.Tanh,output_activation=nn.Identity):
    layers=[]
    for j in range(len(sizes)-1):
        act=activation if j<len(sizes)-2 else output_activation
        layers+=[nn.Linear(sizes[j],sizes[j+1],act())]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v0',hidden_sizes=[32],lr=1e-2,epochs=50,
          batch_size=5000,render=False):
    env=gym.make(env_name)
    assert isinstance(env.observation_space,Box),"This example only works" \
                                                 "for envs with continuous state space."
    assert isinstance(env.action_space,Discrete),\
    "This example only works for envs with discrete action spaces."
    obs_dim=env.observation_space.shape[0]
    n_acts= env.action_space.n

    logits_net=mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    def get_policy(obs):
        logits=logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs,act,weights):
        logp=get_policy(obs).log_prob(act)
        return -(logp*weights).mean()

    optimizer=Adam(logits_net.parameters(),lr=lr)

    def train_one_epoch():
        batch_obs=[]
        batch_acts=[]
        batch_weights=[]
        batch_rets=[]
        batch_lens=[]

        obs=env.reset()
        done=False
        ep_rews=[]

        finished_rendering_this_epoch=False

        while True:
            if (not finished_rendering_this_epoch) and render:
                env.render()

                batch_obs.append(obs.copy())
                act=get_action(torch.as_tensor(obs,dtype=torch.float32))
                obs,rew,done,_=env.step(act)

                batch_acts.append(act)
                ep_rews.append(rew)

                if done:
                    ep_ret,ep_len=sum(ep_rews),len(ep_rews)
                    batch_rets.append(ep_ret)
                    batch_lens.append(ep_len)
                    batch_weights+=[ep_ret]*ep_len
                    obs,done,ep_rews=env.reset(),False,[]
                    finished_rendering_this_epoch=True
                    if len(batch_obs)>batch_size:
                        break

        optimizer.zero_grad()
        batch_loss=compute_loss(obs=torch.as_tensor(batch_obs,dtype=torch.float32),
                                act=torch.as_tensor(batch_acts,dtype=torch.int32),
                                weights=torch.as_tensor(batch_weights,dtype=torch.float32))

        batch_loss.backward()
        optimizer.step()
        return batch_loss,batch_rets,batch_lens

    for i in range(epochs):
        batch_loss,batch_rets,batch_lens=train_one_epoch()
        print('epoch:%3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
              (i,batch_loss,np.mean(batch_rets),np.mean(batch_lens)))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--env_name','--env',type=str,default='CartPole-v0')
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--lr',type=float,default=1e-2)
    args=parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name,render=args.render,lr=args.lr)

