#!/home/muji/mambaforge/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import rospy

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.learning_rate = 0.00025
        self.state_size = state_size
        self.action_size = action_size

        self.layer1 = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.2)
        self.layer3 = nn.Sequential(
            nn.Linear(64, self.action_size)
        )
        self.out = nn.Linear(self.action_size, self.action_size)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.out(x)
        # rospy.loginfo("x: {}".format(x))
        return x
    
    # def criterion(self):
    #     return nn.MSELoss()
    
    # def optimizer(self):
    #     # return optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    #     return optim.RMSprop(model.parameter(), lr=self.learning_rate, alpha=0.9, rho=0.9, epsilon=1e-06)
    

    # def train_loop():
        
    # def test_loop():