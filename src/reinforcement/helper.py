import numpy as np

def get_reward(current_img, goal_img):
    # current_img: (H, W, 3)
    # goal_img: (H, W, 3)
    # return: float
    # compute the difference between the current and goal images
    assert current_img.size == goal_img.size
    # assert current_img.size[2] == 3
    
    diff = np.abs(np.array(current_img) - np.array(goal_img))
    print('mean diff: ', np.mean(diff))
    return np.power(np.mean(diff), -0.01)

def get_goal_img(obs):
    # obs: (H, W, 3)
    # return: (H, W, 3)
    # get the goal image from the obs
    pass