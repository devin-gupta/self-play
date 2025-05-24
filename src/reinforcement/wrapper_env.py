import gymnasium as gym
from gym.spaces import Box, Dict
import numpy as np
from gym.wrappers import TimeLimit
from copy import deepcopy
import gymnasium_robotics

class ImageGoalFrankaWrapper(gym.Env):
    def __init__(self, goal_img, base_env=None, max_steps=200):
        super().__init__()
        self.goal_img = goal_img.astype(np.float32)
        gym.register_envs(gymnasium_robotics)
        self.base_env = base_env or gym.make("FrankaKitchen-v1", render_mode="rgb_array")
        self.base_env = TimeLimit(self.base_env, max_episode_steps=max_steps)

        self.observation_space = Dict({
            "goal_img": Box(0, 255, shape=(480, 480, 3), dtype=np.uint8),
            "curr_img": Box(0, 255, shape=(480, 480, 3), dtype=np.uint8),
        })
        self.action_space = Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.curr_img = np.zeros((480, 480, 3), dtype=np.uint8)

    def _get_image(self):
        # Assumes base_env has render(mode='rgb_array')
        img = self.base_env.render()
        return np.array(img).astype(np.uint8)

    def reset(self):
        self.base_env.reset()
        self.curr_img = self._get_image()
        return {
            "goal_img": self.goal_img,
            "curr_img": self.curr_img,
        }

    def step(self, action):
        obs, _, _, done, info = self.base_env.step(action)
        self.curr_img = self._get_image()

        # Simple pixel-wise reward (e.g. MSE or L2 norm)
        reward = -np.mean((self.curr_img.astype(np.float32) - self.goal_img.astype(np.float32)) ** 2)

        return {
            "goal_img": self.goal_img,
            "curr_img": self.curr_img,
        }, reward, done, info

    def render(self, mode="human"):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
