import numpy as np


def get_reward(current_img, goal_img):
    # current_img: (H, W, 3)
    # goal_img: (H, W, 3)
    # return: float
    # compute the difference between the current and goal images
    assert current_img.shape == goal_img.shape
    assert current_img.shape[2] == 3
    
    diff = np.abs(current_img - goal_img)
    return np.exp(-np.mean(diff))