"""
Policy Gradient Implementation for the CartPole environment
Description:
    Initialize environment [OpenAI gym implementation]
    Agent uses state space information to choose an action
    Update environment state space
    Render environment after the agent becomes certain of it's actions
        (Evaluated by including a Reward Threshold)
Written by: https://github.com/dv-fenix
Date: 19 August 2020
Requirements:
PyTorch 1.6.0
OpenAI Gym 0.8.0
"""

import gym
import matplotlib.pyplot as plt
from torch_agent import PolGrad
import torch.optim as optim

DISPLAY_THRESHOLD = 600
RENDER = False

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

#hyper parameters
GAMMA = 0.99
LR = 0.02

#load agent
model = PolGrad(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    lr=LR,
    reward_decay=GAMMA,
)

optimizer = optim.Adam(model.parameters(), lr=LR)
running_reward = 10

for i_episode in range(3000):
    #observed space
    observation = env.reset()

    while True:
        # Environment rendered after crossing reward threshold
        if RENDER:
            env.render()

        action = model.choose_action(observation)
        done = False

        # load next state
        observation_, reward, done, info = env.step(action)

        model.save_sequence(observation, action, reward)

        if done:
            ep_return = sum(model.ep_rewards)

            running_reward = running_reward * 0.99 + ep_return * 0.01
            if running_reward > DISPLAY_THRESHOLD:
                RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward), " R:",int(ep_return))
            # state-action value
            value = model.pol_update(optimizer)

            if i_episode == 0:
                plt.plot(value)    # plot the episode value
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                # plt.show()
            break
        # update
        observation = observation_