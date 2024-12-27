import gymnasium as gym
import numpy as np

import pygame
import math

from param_loader import ParamLoader
from util import StateEnum as X
from util import normalize_angle
from ode import RK2, RK4
from reward import *

from typing import Any, TypeVar

ObsType = TypeVar("ObsType")

class InvertedPendulumEnv(gym.Env):
    def __init__(self):
        super(InvertedPendulumEnv, self).__init__()

        self.params = ParamLoader()
        self.params = self.params["pendulum"]   

        self.viewer = None
        self.state = None
        self.action = None

        self.action_space = gym.spaces.Box(low=-self.params["max_force"], high=self.params["max_force"], shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.n_step = 0
        self.n_pole = self.params["n_pole"]

    def model_step(self, state, action):
        max_force = self.params["max_force"]
        max_x = self.params["max_x"]
        g = self.params["g"]
        m_cart = self.params["m_cart"]
        m_pole = self.params["m_pole"]
        l = self.params["l"]
        b_cart = self.params["b_cart"]
        b_pole = self.params["b_pole"]
        n_pole = self.n_pole
        M = m_cart + n_pole * m_pole

        u = action * max_force

        theta0 = state[2:2+n_pole]
        d_theta0 = state[2+n_pole:2+2*n_pole]

        cos_theta0 = np.zeros(n_pole)
        sin_theta0 = np.zeros(n_pole)
        for i in range(n_pole):
            cos_theta0[i] = np.cos(theta0[i])
            sin_theta0[i] = np.sin(theta0[i])

        l_hat = l * 0.5

        dd_theta = np.zeros((n_pole,))

        d_x = state[1]
        d_theta = d_theta0
        dd_x = (g * np.sum(m_pole  * sin_theta0 * cos_theta0) - 7.0/3.0 * (u + np.sum(m_pole * l_hat *sin_theta0 * np.square(d_theta0)) - b_cart * d_x) - np.sum(b_pole * d_theta * cos_theta0/l_hat)) / (np.sum(m_pole * np.square(cos_theta0)) - 7.0/3.0 * M)

        dd_theta = (3.0/(7.0 * l_hat)) * (g * sin_theta0 - dd_x * cos_theta0 - (b_pole * d_theta) / (m_pole * l_hat))

        d_state = np.array([d_x, dd_x[0]])
        d_state = np.append(d_state, d_theta)
        d_state = np.append(d_state, dd_theta)
        return d_state
     
    def step(self, action):
        done = False
        self.n_step += 1
        self.action = action
        self.state = RK2(self.model_step, self.state.flatten(), action, self.params["dt"])
        self.state[2:2+self.n_pole] = normalize_angle(self.state[2:2+self.n_pole])
        reward = 0
        if self.n_pole == 1:
            reward = reward_uptime(self.state, action, self.n_step)
        if self.n_pole == 2:
            reward = reward_double(self.state, action, self.n_step)
        if self.n_pole == 3:
            reward = reward_triple(self.state, action, self.n_step)
        if abs(self.state[0]) > 7.5:
            done = True
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        noise = True
        self.n_step = 0
        theta = np.pi
        if options is not None:
            noise = options.get("noise")
        mu_state = np.zeros((2+2*self.n_pole,))
        mu_state[2:2+self.n_pole] = theta
        if noise:
            std_state = np.zeros((2+2*self.n_pole,))
            std_state[1] = 0.0
            std_state[2:2+self.n_pole] = np.pi/4.0
            self.state = np.random.normal(mu_state, std_state)
        else:
            self.state = mu_state
        return self._get_obs(), {}

    def _get_obs(self):
        state = np.zeros((2+2*self.n_pole,), dtype=np.float32)
        for i, s in enumerate(self.state):
            state[i] = np.random.normal(s, 0.0)
        return state

    def render(self, mode='human'):
        width = 1600
        heigth = 500
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((width, heigth))  # Increased height for text
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 30)  # Initialize font

        self.viewer.fill((255, 255, 255))

        # Get state
        x, x_dot = self.state[0:2]
        theta = self.state[2:2+self.n_pole]
        theta_dot = self.state[2+self.n_pole:2+2*self.n_pole]

        # Constants
        SCALE = 100  # pixels per meter
        CART_WIDTH = 50
        CART_HEIGHT = 10 
        POLE_LENGTH = self.params["l"] * SCALE

        # Calculate cart position
        cart_x = int(width // 2 + x * SCALE)
        cart_y = 350

        # Draw cart
        cart_rect = [int(cart_x - CART_WIDTH//2), int(cart_y - CART_HEIGHT//2), CART_WIDTH, CART_HEIGHT]
        pygame.draw.rect(self.viewer, (0, 0, 0), cart_rect) 

        # Calculate pole end position
        pole_x = np.zeros(self.n_pole)
        pole_y = np.zeros(self.n_pole)
        pole_x[0] = int(cart_x + POLE_LENGTH * math.sin(theta[0]))
        pole_y[0] = int(cart_y - POLE_LENGTH * math.cos(theta[0]))

        # Draw pole
        pygame.draw.line(self.viewer, (0, 0, 255), (cart_x, cart_y), (pole_x[0], pole_y[0]), 6)

        for i in range(1, self.n_pole):
            pole_x[i] = int(pole_x[i-1] + POLE_LENGTH * math.sin(theta[i]))
            pole_y[i] = int(pole_y[i-1] - POLE_LENGTH * math.cos(theta[i]))
            pygame.draw.line(self.viewer, (0, 0, 255), (pole_x[i-1], pole_y[i-1]), (pole_x[i], pole_y[i]), 6)


        for i in range(self.n_pole):
            pygame.draw.circle(self.viewer, (0, 0, 255), (int(pole_x[i]), int(pole_y[i])), 10)
            pygame.draw.circle(self.viewer, (255, 0, 0), (int(pole_x[i]), int(pole_y[i])), 10)

        # Draw bob

        # Draw ground
        start_pos = (0, int(cart_y + CART_HEIGHT//2))
        end_pos = (width, int(cart_y + CART_HEIGHT//2))

        pygame.draw.line(self.viewer, (0, 0, 0), start_pos, end_pos, 2) 
        # Render text for state and action
        state_text = f"State: x={float(x):.2f}, x'={float(x_dot):.2f}"
        for i in range(self.n_pole):
            state_text += f", theta_{i}={float(theta[i]):.2f}, theta_{i}'={float(theta_dot[i]):.2f}"
        state_surface = self.font.render(state_text, True, (0, 0, 0))
        self.viewer.blit(state_surface, (10, 410))

        if self.action is not None:
            action_text = f"Action: {float(self.action):.2f}"
            action_surface = self.font.render(action_text, True, (0, 0, 0))
            self.viewer.blit(action_surface, (10, 450))

        pygame.display.flip()
        #self.clock.tick(30)

        if mode == 'rgb_array':
            return pygame.surfarray.array3d(self.viewer).transpose((1, 0, 2))

    def close(self):
        if self.viewer:
            pygame.quit()
            self.viewer = None

if __name__ == "__main__":
    env = InvertedPendulumEnv()
