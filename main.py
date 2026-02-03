import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch import Tensor

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import gymnasium as gym


def main():
    env = gym.make(
        "CartPole-v0",
        render_mode="human",
    )
    for i_episode in range(5):
        state, info = env.reset()
        done = False
        t = 0
        while not done:
            env.render()
            t += 1
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(f"Episode finished after {t} timesteps")
    env.close()


class PolicyNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.train()  # set training mode

    def forward(self, x):
        # TODO: Define the forward pass of your model

        pass


class Reinforce:
    def __init__(self, n_states, n_actions):
        self.gamma = 0.99
        self.lr = 1e-4
        self.n_actions = n_actions
        self.n_states = n_states
        self.model = PolicyNet(n_states, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.onpolicy_reset()

    def act(self, state):
        # TODO: Implement the method that takes a state as input
        # and uses your PolicyNet model to return / sample actions
        # your agent should perform in the environment. You need to
        # sample from `Categorical`. Read the corresponding
        # documentation if you have never used it before.
        #
        ########################################################
        pass

    def onpolicy_reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def update_policy(self):
        actions = torch.tensor(self.actions, dtype=torch.float32)
        rewards = np.array(self.rewards)
        T = len(rewards)
        R = np.zeros_like(rewards)
        loss = torch.tensor(0.0)
        # TODO: your code to update the policy using PyTorch
        # - you need to compute the discounted rewards, i.e. R[t] for t in range(T)
        # - use the rewards together with the log-probs to compute the loss
        # - differentiation of the PolicyNet is handled by PyTorch for you
        ########################################################
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


if __name__ == "__main__":
    main()
