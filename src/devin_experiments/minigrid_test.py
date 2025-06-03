import minari
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import gymnasium as gym

dataset = minari.load_dataset('minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0')
# Filter for episodes that are not too short and not too long
print(f'TOTAL EPISODES ORIGINAL DATASET: {dataset.total_episodes}')
filter_dataset = dataset.filter_episodes(lambda episode: (len(episode.actions) > 5) and (len(episode.actions) < 30) and (episode.rewards.mean() > 0))
print(f'TOTAL EPISODES FILTER DATASET: {filter_dataset.total_episodes}')
env = filter_dataset.recover_environment(render_mode='human')
filter_dataset.set_seed(seed=1)

max_steps = 30

for episode in filter_dataset.sample_episodes(n_episodes=5):  # Let's look at just 5 episodes for clarity
    print(f"\nStarting new episode with {len(episode.actions)} actions")
    print(f"Original episode mean reward: {episode.rewards.mean():.3f}")
    print(f"Original episode rewards: {episode.rewards}")
    
    # Reset environment and get initial observation
    obs, info = env.reset()
    truncated, terminated = False, False
    
    replay_rewards = []
    # Use the episode's recorded actions
    for step, action in enumerate(episode.actions):
        if step >= max_steps:
            break
            
        obs, reward, truncated, terminated, info = env.step(action)
        replay_rewards.append(reward)
        print(f"Step {step}: Action {action}, Reward {reward:.2f}")
        time.sleep(0.1)
        env.render()

        if truncated or terminated:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Replay mean reward: {np.mean(replay_rewards):.3f}")
    print(f"Replay rewards: {replay_rewards}")

env.close()