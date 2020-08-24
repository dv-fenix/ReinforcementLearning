"""
Advantage Actor Critic Implementation for the CartPole environment
This file contains the code to model the A2C agent.
Written by: https://github.com/dv-fenix
Date: 21 August 2020
Requirements:
PyTorch 1.6.0
OpenAI Gym 0.8.0
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.init as init

#set seed to replicate results
np.random.seed(1)
torch.manual_seed(1)
torch.autograd.set_detect_anomaly(True)

class ActorCritic(nn.Module):
    def __init__(self, n_inputs: int, n_actions: int,
                        n_features: int=64, discount_factor: float=0.99):
        """ Advantage Actor Critic Agent
        Args:
            :param n_inputs:    integer representing observation/state size
            :param n_actions:   integer representing size of action space
            :param n_features:  hidden_dim of MLP
            :param learning_rate:   learning rate for the agent

        Methods:
            private:
                _advantage(Q_value) ---> Float
                    It computes and returns the advantage score for the episode
            public:
                forward(observation) ---> action-value, action-distribution
                    Forward pass over the computational graph
                choose_action(aobservation) ---> action-value, action
                    Samples the action from action distribution
                save_sequence(value, action, reward) ---> None
                    Saves the sequence for the current time-step
                update(optimizer, Q_value) ---> advantage
                    Executes backpropagation over the Computational Graph
                    Updates the policy
                    Resets the Agent for next episode

        """
        #Constructor
        super(ActorCritic, self).__init__()

        self.num_actions = n_actions
        self.gamma = discount_factor
        self.dropout = nn.Dropout(0.8)
        self.entropy = 0
        # instantiate critic layers
        self.critic_layer1 = nn.Linear(n_inputs, n_features, bias=False)
        self.critic_layer2 = nn.Linear(n_features, self.num_actions, bias=False) # number of values generated = possible actions

        # instantiate actor layers
        self.actor_layer1 = nn.Linear(n_inputs, n_features, bias=False)
        self.actor_layer2 = nn.Linear(n_features, self.num_actions, bias=False)

        # instantiate activations
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        # initialize weights
        init.xavier_normal_(self.critic_layer1.weight)
        init.xavier_normal_(self.critic_layer2.weight)

        init.xavier_normal_(self.actor_layer1.weight)
        init.xavier_normal_(self.actor_layer2.weight)

        # current state of agent in an episode
        self.ep_obs, self.ep_as, self.ep_rewards, self.ep_values = [], [], [], []
        self.ep_policy = Variable(torch.Tensor())

    def forward(self, observation):
        # find state-value
        value_layer1 = self.relu(self.critic_layer1(observation))
        value = self.critic_layer2(value_layer1)

        # find action prob_dist
        prob_layer1 = self.relu(self.actor_layer1(observation))
        actions_dist = self.softmax(self.actor_layer2(prob_layer1))

        return value, actions_dist

    def choose_action(self, observation):
        obs = torch.from_numpy(observation).type(torch.FloatTensor)
        value, act = self.forward(Variable(obs))
        # pi(at|st) dist, categorical wrapper
        prob = Categorical(act)
        # Sampling actions from policy distribution
        action = prob.sample()
        # store policy history
        if self.ep_policy.dim() != 0:
            self.ep_policy = torch.cat([self.ep_policy, prob.log_prob(action).unsqueeze(0)])
        else:
            self.ep_policy = (prob.log_prob(action))
        action = action.numpy()

        # entropy term for loss function, no backprop [constant]
        dist = act.detach()
        entropy = -torch.sum(dist.mean() * torch.log(dist))
        self.entropy += entropy
        return value, action

    def save_sequence(self, observation, action, reward, value):
        self.ep_obs.append(observation)
        self.ep_as.append(action)
        self.ep_rewards.append(reward)
        self.ep_values.append(value)

    def update(self, optimizer, Q_value):

        advantage = self._advantage(Q_value)
        actor_loss = (torch.sum(torch.mul(self.ep_policy, Variable(advantage)).mul(-1), -1))
        critic_loss = 0.5 * advantage.pow(2).mean()

        # total loss
        a2c_loss = actor_loss + critic_loss + 0.001 * self.entropy


        optimizer.zero_grad()
        a2c_loss.backward(retain_graph=True)
        optimizer.step()

        # reset agent for next episode
        self.ep_obs, self.ep_as, self.ep_rewards, self.ep_values = [], [], [], []
        self.ep_policy = Variable(torch.Tensor())


        return advantage

    def _advantage(self, Q_value):

        Q_values = []
        values = []
        # reversed rewards
        for r in self.ep_rewards[::-1]:
            # Q_(s_t, a_t)
            Q_value = r + self.gamma * Q_value
            Q_values.insert(0, Q_value)
        q_values = torch.FloatTensor(Q_values)
        # detach and store current state values according to time step
        for element in self.ep_values:
            values.insert(0, element.detach())
        values = torch.FloatTensor(values)

        # Advantage = r_(t+1) + gamma* V_(s_t+1) - V_(s_t)
        advantage = q_values - values

        return advantage