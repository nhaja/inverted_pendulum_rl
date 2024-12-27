import torch as nn
import torch.nn.functional as F
import numpy as np

from actor import Actor
from critic import Critic

from param_loader import ParamLoader

import random

device = nn.device("cuda:0" if nn.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_dim, action_dim, actor, critic, actor_target, critic_target, replay_buffer, gamma=0.99, tau=0.001, batch_size=64, num_agents=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._actor = actor
        self._critic = critic
        self._actor_target = actor_target
        self._critic_target = critic_target
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_agents = num_agents

        self.actor_optimizer = nn.optim.Adam(self._actor.parameters(), lr=0.0001)
        self.critic_optimizer = nn.optim.Adam(self._critic.parameters(), lr=0.001)

        self.params = ParamLoader()
        self.pendulum_params = self.params["pendulum"]
        self.max_force = self.pendulum_params["max_force"]

    def step(self, states, actions, rewards, next_states, dones):
        self.replay_buffer.add(states, actions, rewards, next_states, dones)

        if len(self.replay_buffer) > self.batch_size:
            experiences = self.replay_buffer.sample()
            self._train(experiences)

    def act(self, state, add_noise=True):
        state = nn.from_numpy(state).float().to(device)
        acts = np.zeros((self.num_agents, self.action_dim))
        self._actor.eval()
        with nn.no_grad():
            acts = self._actor(state).cpu().data    
        self._actor.train()
        if add_noise:
            acts += nn.distributions.Normal(0, 0.1).sample().numpy()
        return np.clip(acts, -1.0, 1.0)

    def _train(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        actions_next = self._actor_target(next_states)
        Q_targets_next = self._critic_target(next_states, actions_next)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self._critic(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self._actor(states)
        actor_loss = -self._critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self._critic_target, self._critic)
        self._soft_update(self._actor_target, self._actor)

    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            updated_param = self.tau * param.data + (1 - self.tau) * target_param.data
            target_param.data.copy_(updated_param)

