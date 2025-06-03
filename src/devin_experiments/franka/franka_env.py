import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave'], render_mode='human')

env.reset()
for _ in range(50):
    action = env.action_space.sample()  # Sample a random action
    env.step(action)  # Take the action in the environment
    env.render()  # Render the environment

env.close()  # Close the environment when done