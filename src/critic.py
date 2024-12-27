import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ) 
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                fan_in = layer.weight.data.size()[0]
                lim = 1. / np.sqrt(fan_in)
                layer.weight.data.uniform_(-lim, lim)
        

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.model(x)
 
