import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch

class CustomFrankaEnv(gym.Wrapper):
    """
    A custom wrapper for the FrankaKitchen-v1 environment to:
    1. Define a custom reward function.
    2. Flatten the observation space to be compatible with Stable Baselines3 (SAC).
    """
    def __init__(self, env):
        super().__init__(env)
        self.base_env = env
        self.observation_space = spaces.Box(
            low = -1.0, high = 1.0, shape = (480, 480), dtype = np.float32
        )
        self.action_space = self.base_env.action_space

    def get_curr_image(self):
        img = Image.fromarray(self.base_env.render())
        img = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=1)
        img = torchvision.transforms.functional.resize(img, (120, 120))
        return np.array(img, dtype = np.float32)
    
    def get_goal_image(self):
        img = Image.open('goal_image.png')
        img = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=1)
        img = torchvision.transforms.functional.resize(img, (120, 120))
        return np.array(img, dtype = np.float32)
    
    def generate_observation(self):
        curr_img = self.get_curr_image()
        goal_img = self.get_goal_image()
        assert curr_img.shape == goal_img.shape, 'Current image and goal image must have the same shape, but got {} and {}'.format(curr_img.shape, goal_img.shape)
        diff_img = np.abs(curr_img - goal_img)
        return diff_img

    def step(self, action):
        observation_dict, original_reward, terminated, truncated, info = self.base_env.step(action)

        original_reward -= 0.1

        kettle_dist = np.linalg.norm(observation_dict['achieved_goal']['kettle'] - observation_dict['desired_goal']['kettle'])
        if abs(kettle_dist - self.default_kettle_dist) > 0.0001:
            original_reward += 1
            print('kettle moved')

        return self.generate_observation(), original_reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation_dict, info = self.base_env.reset(**kwargs)

        # store default kettle attributes for later stepping
        self.default_kettle_dist = np.linalg.norm(observation_dict['achieved_goal']['kettle'] - observation_dict['desired_goal']['kettle'])
        self.default_kettle_state = observation_dict['achieved_goal']['kettle']

        return self.generate_observation(), info

    def render(self):
        render_img = self.generate_observation()
        plt.imshow(render_img)
        plt.pause(0.01)
        return self.base_env.render()

    def close(self):
        plt.close()
        return self.base_env.close()