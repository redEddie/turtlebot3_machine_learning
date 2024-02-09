import os
import dill
import random
from collections import deque, namedtuple


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple(
            "Transition", ("state", "action", "reward", "next_state")
        )

    def __len__(self):
        return len(self.memory)

    def pop(self):
        return self.memory.pop()

    def push(self, *args):
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save(self, dir_path, episode):
        save_path = dir_path + "/memory" + str(episode) + ".pkl"

        with open(save_path, "wb") as outfile:
            dill.dump(self.memory, outfile)
        print("[MEMORY] Memory Save Success @{}".format(episode))
        print(
            "[MEMORY] Memory Length: {}, Capacity: {}".format(
                len(self.memory), self.capacity
            )
        )

    def load(self, dir_path, file_name):
        load_path = dir_path + "/" + file_name + ".pkl"

        with open(load_path, "rb") as infile:
            self.memory = dill.load(infile)
        print("[MEMORY] Memory Load Success ({})".format(len(self.memory)))
