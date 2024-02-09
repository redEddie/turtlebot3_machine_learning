from collections import namedtuple, deque
import dill
import random
import os, time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .env import Env


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )

    def push(self, *args):
        """transition 저장"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_data_as_tensors(self):
        states, actions, next_states, rewards = zip(
            *[(t.state, t.action, t.next_state, t.reward) for t in self.memory]
        )

        states = states.view(len(states), -1)
        actions = actions.view(len(actions), -1)
        next_states = next_states.view(len(next_states), -1)

        return (
            torch.tensor(states),
            torch.tensor(actions),
            torch.tensor(next_states),
            torch.tensor(rewards),
        )

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    # in = 30, out = 5
    def __init__(self, state_size, action_size):
        super().__init__()

        self.in_layer = nn.Linear(state_size, 64)
        self.h_layer = nn.Linear(64, 64)
        self.out_layer = nn.Linear(64, action_size)

        self.act = nn.ReLU()

    def forward(self, x):
        h1 = self.act(self.in_layer(x))
        h2 = self.act(self.h_layer(h1))
        output = self.out_layer(h2)
        return output


class DQN_EXPERT:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.s_dim = 30
        self.a_dim = 5
        self.model = DQN(self.s_dim, self.a_dim).to(self.device)

        self.env = Env(action_size=self.a_dim)
        self.memory = ReplayMemory(6400)

        self.epochs = 30
        self.max_iter = 150

    def load_expert(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)

    def get_action(self, state):
        self.model.eval()
        return self.model(state).max(1)[1].view(1, 1)

    def get_expert_history(self):
        for epoch in range(self.epochs):
            score = 0

            state = self.env.reset()
            state = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            for step in range(self.max_iter):
                action = self.get_action(state)
                next_state, reward, truncated = self.env.step(action)
                score += reward
                self.memory.push(state, action, next_state, reward)
                state = next_state
                state = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )

                if truncated:
                    break
                if reward == 500:
                    break
                if step == self.max_iter and reward != 500:
                    reward = -100

            print("[expert dqn] epoch: {:2d}, score: {:4f}".format(epoch, score))

        index = time.localtime()
        index = "{:02d}_{:02d}_{:02d}_{:02d}".format(
            index.tm_mon, index.tm_mday, index.tm_hour, index.tm_min
        )

        folder = os.path.dirname(os.path.realpath(__file__))
        folder = folder.replace("script/qldf", "node")
        file = folder + "/expert_history.pkl"
        with open(file, "wb") as outfile:
            dill.dump(self.memory, outfile)
        print(
            "[expert dqn] expert history has been saved(len: {})".format(
                len(self.memory)
            )
        )
