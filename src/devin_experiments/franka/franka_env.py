import gymnasium as gym
import gymnasium_robotics
from utils.custom_franka_env import CustomFrankaEnv

gym.register_envs(gymnasium_robotics)

env = CustomFrankaEnv(gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode='human'))

obs, info = env.reset()
for _ in range(50):
    action = env.action_space.sample()  # Sample a random action
    obs, reward, terminated, truncated, info = env.step(action)  # Take the action in the environment
    print('reward: ', reward)
    env.render()  # Render the environment

env.close()  # Close the environment when done