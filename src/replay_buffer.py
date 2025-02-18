import torch as nn
import numpy as np
import random
from collections import namedtuple, deque

device = nn.device("cuda:0" if nn.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = nn.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = nn.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = nn.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = nn.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = nn.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
