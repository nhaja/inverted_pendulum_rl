import gymnasium as gym
from gymnasium.envs.registration import register

import torch as nn
import numpy as np

from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer
from ddpg_agent import DDPGAgent
from param_loader import ParamLoader

from tqdm import tqdm

import os

render_train = True
render_val = True
batch_size = 128

device = nn.device("cuda:0" if nn.cuda.is_available() else "cpu")

params = ParamLoader()
gamma = params["model"]["gamma"]
max_force = params["pendulum"]["max_force"]

def tensor(x):
    return nn.tensor(x).float()

print("Registering InvertedPendulum-v0 environment...")

register(
    id='cartpole_custom',
    entry_point='inverted_pendulum:InvertedPendulumEnv',
    max_episode_steps=1000,
)


env = gym.make("cartpole_custom")
state_dim = 2 + 2 * params["pendulum"]["n_pole"]
action_dim = 1
print("Environment Registered:")
print(f"State Dimension: {state_dim}")
print(f"Action Dimension: {action_dim}")

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)
actor_target = Actor(state_dim, action_dim)
critic_target = Critic(state_dim, action_dim)

save_dir = f"results/pendulum{params["pendulum"]["n_pole"]}"

if os.path.exists(save_dir+"/actor_weights.pth"):
    actor.load_state_dict(nn.load(save_dir+"/actor_weights.pth", weights_only=True))
    critic.load_state_dict(nn.load(save_dir+"/critic_weights.pth", weights_only=True))
    actor_target.load_state_dict(nn.load(save_dir+"/actor_target_weights.pth", weights_only=True))
    critic_target.load_state_dict(nn.load(save_dir+"/critic_target_weights.pth", weights_only=True))

replay_buffer = ReplayBuffer(action_dim, 1000, 128)
agent = DDPGAgent(
    state_dim=state_dim, 
    action_dim=action_dim,
    actor=actor, 
    critic=critic, 
    actor_target=actor_target, 
    critic_target=critic_target, 
    replay_buffer=replay_buffer,
    batch_size=batch_size
)

episodes = 50000
validation_episodes = 50
episode_rewards = []
max_steps = 100

pbar_train = tqdm(range(episodes), desc='Training Progress')
pbar_val = tqdm(range(episodes//validation_episodes), desc='Validation Progress')
for episode in pbar_train:
    options = {"noise": True}
    state, _ = env.reset(options=options)
    agent_reward = 0
    
    for step in range(max_steps):
        if render_train:
            env.render()
        action = agent.act(state)

        next_state, reward, done, _, _ = env.step(action.detach().numpy())
        agent.step(state, action, reward, next_state, done)

        state = next_state
        agent_reward += reward
        
        if done:
            break
    

    if episode % 50 == 0 and episode != 0:
        actor.eval()
        val_rewards = []
        for val_episode in range(10):
            options = {"noise": False}
            state, _ = env.reset(options=options)

            val_reward = 0
            for step in range(max_steps):
                if render_val:
                    env.render()
                action = agent.act(state, add_noise=False)
                next_state, reward, done, _, _ = env.step(action.detach().numpy())
                state = next_state
                val_reward += reward
                if done:
                    break

            val_rewards.append(val_reward)
            pbar_val.set_postfix({"Average Validation Reward": f"{np.mean(val_rewards):.2f}"})
        pbar_val.update(1)

    episode_rewards.append(agent_reward)
    # safe weights every 100 episodes
    if episode % 50 == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        nn.save(actor.state_dict(), save_dir+"/actor_weights.pth")
        nn.save(critic.state_dict(), save_dir+"/critic_weights.pth")
        nn.save(actor_target.state_dict(), save_dir+"/actor_target_weights.pth")
        nn.save(critic_target.state_dict(), save_dir+"/critic_target_weights.pth")
    # Update the progress bar description with the latest reward
    pbar_train.set_postfix({"Episode Reward": f"{agent_reward:.2f}"})

env.close()


