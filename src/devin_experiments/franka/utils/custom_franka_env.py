import gymnasium as gym
import numpy as np

class CustomFrankaEnv(gym.Wrapper):
    """
    A custom wrapper for the FrankaKitchen-v1 environment to define a custom reward function.
    """
    def __init__(self, env):
        """
        Initializes the wrapper with the base environment.

        Args:
            env: The base Gymnasium environment (e.g., FrankaKitchen-v1 instance).
        """
        super().__init__(env)
        self.base_env = env # Store a reference to the base environment if needed

    def step(self, action):
        """
        Performs a step in the environment with the given action and calculates a custom reward.

        Args:
            action: The action to take in the environment.

        Returns:
            observation (object): Agent's observation of the current environment.
            reward (float): Amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended due to reaching a terminal state.
            truncated (bool): Whether the episode has ended due to a time limit or other truncation.
            info (dict): Contains auxiliary diagnostic information (useful for debugging, learning).
        """
        # Take a step in the base environment
        observation, original_reward, terminated, truncated, info = self.base_env.step(action)

        original_reward -= 0.01  # Small penalty for each step to encourage efficiency

        kettle_position = observation['observation'][32:35]
        # Penalty for distance from kettle's target position
        kettle_target_position = np.array([-0.23, 0.75, 1.62])
        original_reward -= np.linalg.norm(kettle_position - kettle_target_position) * 0.1

        if info.get('kettle_task_success', False):
            original_reward += 100.0 # Big reward for success!
            print("Kettle task completed! Giving large reward.") # Corrected print statement

        return observation, original_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Resets the base environment.
        """
        return self.base_env.reset(**kwargs)

    def render(self):
        """
        Renders the base environment.
        """
        return self.base_env.render()

    def close(self):
        """
        Closes the base environment.
        """
        return self.base_env.close()
