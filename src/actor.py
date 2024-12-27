import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        ) 
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                fan_in = layer.weight.data.size()[0]
                lim = 1. / np.sqrt(fan_in)
                layer.weight.data.uniform_(-lim, lim)

        self.model[-2].weight.data.uniform_(-3e-3, 3e-3)
        

    def forward(self, state):
        return self.model(state)

