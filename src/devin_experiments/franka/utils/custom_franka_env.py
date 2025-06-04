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
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low = 0, high = 255, shape = (1, 120, 120), dtype = np.uint8
            ),
            'achieved_goal': spaces.Box(
                low = 0, high = 255, shape = (1, 120, 120), dtype = np.uint8
            ),
            'desired_goal': spaces.Box(
                low = 0, high = 255, shape = (1, 120, 120), dtype = np.uint8
            ),
        })
        self.action_space = self.base_env.action_space

    def get_curr_image(self):
        img = Image.fromarray(self.base_env.render())
        img = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=1)
        img = torchvision.transforms.functional.resize(img, (120, 120))
        return np.array(img, dtype = np.uint8)
    
    def get_goal_image(self):
        img = Image.open('goal_image.png')
        img = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=1)
        img = torchvision.transforms.functional.resize(img, (120, 120))
        return np.array(img, dtype = np.uint8)
    
    def generate_observation(self):
        curr_img = self.get_curr_image()
        goal_img = self.get_goal_image()
        assert curr_img.shape == goal_img.shape, 'Current image and goal image must have the same shape, but got {} and {}'.format(curr_img.shape, goal_img.shape)
        diff_img = np.expand_dims(np.abs(curr_img - goal_img), axis=0)

        observation = {
            'observation': np.expand_dims(np.abs(curr_img - goal_img), axis=0),
            'achieved_goal': np.expand_dims(np.abs(curr_img), axis=0),
            'desired_goal': np.expand_dims(np.abs(goal_img), axis=0)
        }

        if self.observation_space.contains(observation):
            return observation
        else:
            print('Observation is not in the observation space')
            exit()

    def compute_reward(self, achieved_goal, desired_goal, info):

        # print('\n Achieved goal shape: ', achieved_goal.shape)

        diff = achieved_goal - desired_goal

        if diff.shape == (1, 120, 120):
            return -np.linalg.norm(diff)
        else:
            return -np.linalg.norm(diff[:, 0, :, :], axis=(1, 2))



    def step(self, action):
        observation_dict, original_reward, terminated, truncated, info = self.base_env.step(action)

        observation = self.generate_observation()

        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation_dict, info = self.base_env.reset(**kwargs)

        # store default kettle attributes for later stepping
        self.default_kettle_dist = np.linalg.norm(observation_dict['achieved_goal']['kettle'] - observation_dict['desired_goal']['kettle'])
        self.default_kettle_state = observation_dict['achieved_goal']['kettle']

        return self.generate_observation(), info

    def render(self, vis=False):
        render_img = self.generate_observation()

        if vis:
            plt.imshow(render_img[0], cmap='jet')
            plt.pause(0.01)
        return self.base_env.render()

    def close(self):
        plt.close()
        return self.base_env.close()