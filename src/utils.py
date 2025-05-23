import matplotlib.pyplot as plt
import minari
import numpy as np
import torch



dataset = minari.load_dataset('D4RL/kitchen/partial-v2')
#dataset.set_seed(seed=1)

env = dataset.recover_environment(render_mode='human')

episode = dataset.sample_episodes(n_episodes=1)[0]

obs = env.reset()

for t in range(episode.actions.shape[0]):
    env.render()

    obs, reward, terminated, truncated, info = env.step(episode.actions[t])

    if terminated or truncated:
        break

    print(f'tasks_to_complete: {info["tasks_to_complete"]}')
    print(f'step_task_completions: {info["step_task_completions"]}')
    print(f'episode_task_completions: {info["episode_task_completions"]}')

print(episode.actions.shape)


env.close()