"""
Policy Gradient Implementation for the CartPole environment
This file contains the code to model the vanilla Policy Gradient agent.
Written by: https://github.com/dv-fenix
Date: 19 August 2020
Requirements:
PyTorch 1.6.0
OpenAI Gym 0.8.0
"""


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical


np.random.seed(1)
torch.manual_seed(1)

class PolGrad(nn.Module):

    def __init__(self, n_actions: int, n_features: int,
                 lr: float=0.01, reward_decay: float=0.99, output_graph: bool=False):
        #Constructor
        super(PolGrad, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = reward_decay
        self.dense1 = nn.Linear(self.n_features, 64, bias=False)
        self.dense2 = nn.Linear(64, self.n_actions, bias=False)
        self.drop = nn.Dropout(0.6)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()
        self.ep_obs, self.ep_as, self.ep_rewards = [], [], []
        self.ep_policy = Variable(torch.Tensor())

    def forward(self, x):
        # simple MLP for policy distribution
        layer_1 = self.activation(self.dense1(x))
        layer_1 = self.drop(layer_1)

        output = self.softmax(self.dense2(layer_1))
        return output

    def choose_action(self, observation):
        obs = torch.from_numpy(observation).type(torch.FloatTensor)
        obs = self.forward(Variable(obs))
        # pi(at|st)
        prob = Categorical(obs)
        # Sampling actions from policy distribution
        action = prob.sample()
        # store policy history
        if self.ep_policy.dim() != 0:
            self.ep_policy = torch.cat([self.ep_policy, prob.log_prob(action).unsqueeze(0)])
        else:
            self.ep_policy = (prob.log_prob(action))
        action = action.numpy()
        return action

    def save_sequence(self, observation, action, reward):
        self.ep_obs.append(observation)
        self.ep_as.append(action)
        self.ep_rewards.append(reward)

    def pol_update(self, optimizer):

        discounted_episode_reward = self._discounted_episode_rewards()
        loss = (torch.sum(torch.mul(self.ep_policy, Variable(discounted_episode_reward)).mul(-1), -1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # reset agent for next episode
        self.ep_obs, self.ep_as, self.ep_rewards = [], [], []
        self.ep_policy = Variable(torch.Tensor())

        return discounted_episode_reward

    def _discounted_episode_rewards(self):
        running_sum = 0
        rewards = []
        # reversed rewards
        for r in self.ep_rewards[::-1]:
            running_sum = r + self.gamma * running_sum
            rewards.insert(0, running_sum)

        rewards = torch.FloatTensor(rewards)
        # normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        return rewards


