import minari
import time
import numpy as np
import matplotlib.pyplot as plt

# https://minari.farama.org/datasets/D4RL/kitchen/mixed-v2/
dataset = minari.load_dataset('D4RL/kitchen/mixed-v2')
dataset.set_seed(seed=1)
print("Observation space:", dataset.observation_space)
print("Action space:", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)

env  = dataset.recover_environment(render_mode='human')

episode = dataset.sample_episodes(n_episodes=1)[0]
obs = env.reset()

for t in range(100):
    env.render()

    # obs, reward, terminated, truncated, info = env.step(env.action_space.sample()) # random action
    obs, reward, terminated, truncated, info = env.step(episode.actions[t]) # dataset action

    if terminated or truncated:
        break

    if reward > 0:
        print(f"Action {t} gave us the following reward: {reward}")

print(obs)
env.close()