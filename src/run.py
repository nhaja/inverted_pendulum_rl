import gymnasium as gym
from gymnasium.envs.registration import register

import torch as nn

from actor import Actor
from param_loader import ParamLoader
import os
import time
import numpy as np

def tensor(x):
    return nn.from_numpy(x).cpu().data

if __name__ == '__main__':
    print("Registering InvertedPendulum-v0 environment...")
    
    register(
        id='InvertedPendulum-v0',
        entry_point='inverted_pendulum:InvertedPendulumEnv',
        max_episode_steps=1000,
    )
    
    env = gym.make('InvertedPendulum-v0')

    params = ParamLoader()
    params = params["pendulum"]

    state_dim = 2 + 2 * params["n_pole"]
    action_dim = 1
            
    print("Environment Registered:")
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")

    actor = Actor(state_dim, action_dim)
    save_dir = f"results/pendulum{params["n_pole"]}"
    if os.path.exists(save_dir):
        actor.load_state_dict(nn.load(save_dir+"/actor_weights.pth", weights_only=True))
        
    max_steps = 10000
    options = {"noise": True}
    state, _ = env.reset(options=options)
    print("Press any key to start simulation")
    env.render()
    input()
    for step in range(max_steps):
        print(state)
        env.render()
        start = time.time()
        action = actor(tensor(state))
        state, reward, done, _, _ = env.step(action.detach().numpy())
        #action = np.array([0.0])
        #state, reward, done, _, _ = env.step(action)

        end = time.time()
        time.sleep(max(params["dt"] - (end - start), 0))
