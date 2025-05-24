from wrapper_env import ImageGoalFrankaWrapper

import minari

dataset = minari.load_dataset('D4RL/kitchen/mixed-v2')
bc_env = dataset.recover_environment(render_mode='rgb_array')
dataset.set_seed(seed=1)

episode = dataset.sample_episodes(n_episodes=1)[0]
bc_env.reset()
initial_obs = bc_env.render()

for t in range(30):
    _, _, _, _, _ = bc_env.step(episode.actions[t])

goal_img = bc_env.render()



from matplotlib import pyplot as plt
import time
import cv2 



goal_image = goal_img  # Load or generate a 480x480x3 goal image
env = ImageGoalFrankaWrapper(goal_img=goal_image)
obs = env.reset()
curr_img = obs['curr_img']

for t in range(30):

    overlay = cv2.addWeighted(curr_img, 0.5, goal_img, 0.5, 0)
    cv2.imshow('New Image', overlay)
    cv2.waitKey(100)

    obs, reward, done, info = env.step(env.action_space.sample())
    curr_img = obs['curr_img']


cv2.destroyAllWindows()
