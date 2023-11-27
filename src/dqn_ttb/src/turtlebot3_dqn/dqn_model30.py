#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    # in = 30, out = 5
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # self.learning_rate = 0.00025
        self.layer1 = nn.Sequential(nn.Linear(state_size, 240), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(240, 240), nn.ReLU())
        # self.dropout = nn.Dropout(0.2)
        self.layer3 = (
            nn.Sequential(  ## 마지막 레이어는 ReLu를 사용하지 않는다. 사용한다면 출력값이 0~1 사이로 제한된다.
                nn.Linear(240, action_size)
            )
        )
        self.out = nn.Linear(action_size, action_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.dropout(x)
        x = self.layer3(x)
        # x = self.out(x)       # 이게 없으니까 발산은 덜한다.
        return x
