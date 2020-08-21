"""
Advantage Actor Critic Implementation for the CartPole environment
Description:
    Initialize environment [OpenAI gym implementation]
    Agent uses state space information to choose an action
        Actor evaluates action distribution
        Critic evaluates Q_values for advantage
    Update environment state space according to gradients influenced by advantage
    Render environment after the agent becomes certain of it's actions
        (Evaluated by including a Return Threshold)
Written by: https://github.com/dv-fenix
Date: 21 August 2020
Requirements:
PyTorch 1.6.0
OpenAI Gym 0.8.0
"""

import gym
import matplotlib.pyplot as plt
from torch_agent_a2c import ActorCritic
import torch.optim as optim
from torch.autograd import Variable
import torch

DISPLAY_THRESHOLD = 400
RENDER = False

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

# hyper parameters
GAMMA = 0.99
LR = 0.01

# load agent
model = ActorCritic(
    n_actions=env.action_space.n,
    n_inputs=env.observation_space.shape[0],
    discount_factor=GAMMA,
    n_features=32
)

optimizer = optim.Adam(model.parameters(), lr=LR)
running_reward = 10

for i_episode in range(3000):
    # observed space
    observation = env.reset()

    while True:
        # Environment rendered after crossing return threshold
        if RENDER:
            env.render()

        value, action = model.choose_action(observation)
        done = False

        # load next state
        observation_, reward, done, info = env.step(action)
        # save current state-action sequence
        model.save_sequence(observation, action, reward, value)

        if done:
            ep_return = sum(model.ep_rewards)
            running_reward = running_reward * 0.99 + ep_return * 0.01
            if running_reward > DISPLAY_THRESHOLD:
                RENDER = True     # rendering
            print("episode:", i_episode, "  return:", int(running_reward), " R:",int(ep_return))

            # convert new observation to tensor
            obs = torch.from_numpy(observation_).type(torch.FloatTensor)
            # state-action value for new observation [V_(t+1)]
            Q_value, _ = model.forward(Variable(obs))
            # do not perform gradient update owing to operations on Q_value
            Q_value = Q_value.detach().numpy()[0]
            # update agent
            advantage = model.update(optimizer, Q_value)


            if i_episode == 0:
                plt.plot(advantage)    # plot the episode value
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                # plt.show()
            break
        # update
        observation = observation_