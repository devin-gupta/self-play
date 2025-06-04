import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium import spaces

# Register the environment
# gym.register_envs(gymnasium_robotics)


# --- Custom Environment Wrapper ---
class CustomFrankaEnv(gym.Wrapper):
    """
    A custom wrapper for the FrankaKitchen-v1 environment to:
    1. Define a custom reward function.
    2. Flatten the observation space to be compatible with Stable Baselines3 (SAC).
    """
    def __init__(self, env):
        super().__init__(env)
        self.base_env = env # Store a reference to the base environment

        # Define the new observation space based on 'observation' key
        # Assuming 'observation' key holds a Box space.
        # Access the observation space of the *unwrapped* environment to get the original structure
        original_obs_space = self.base_env.observation_space['observation']
        self.observation_space = spaces.Box(
            low=original_obs_space.low,
            high=original_obs_space.high,
            shape=original_obs_space.shape,
            dtype=original_obs_space.dtype
        )
        print(f"CustomFrankaEnv: New observation space shape: {self.observation_space.shape}")


    def step(self, action):
        observation_dict, original_reward, terminated, truncated, info = self.base_env.step(action)
        observation = observation_dict['observation']

        kettle_dist = np.linalg.norm(observation_dict['achieved_goal']['kettle'] - observation_dict['desired_goal']['kettle'])
        original_reward -= kettle_dist

        if kettle_dist < 0.05:  # Assuming a threshold for success
            original_reward += 100.0 # Big reward for success!
            print("Kettle task completed! Giving large reward.")

        return observation, original_reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation_dict, info = self.base_env.reset(**kwargs)
        # Return only the 'observation' array
        return observation_dict['observation'], info

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()