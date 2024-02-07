#!/home/muji/mambaforge/bin/python

import random
from collections import deque, namedtuple
import torch
import rospy

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

# agent.memory.push(state, action, reward, next_state, done)
    def push(self, *args):
        # rospy.loginfo(args)
        self.memory.append(Transition(*args))
        # rospy.loginfo("memory: {}".format(self.memory))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        # memory_list = list(self.memory)
        # indices = torch.randperm(len(memory_list))[:batch_size]
        # mini_batch = [memory_list[i] for i in indices]
        # return mini_batch

    def __len__(self):
        return len(self.memory)
    
    def print(self):
        return self.memory.state