import numpy as np
from PIL import Image

def get_reward_v1(current_img, goal_img):
    # current_img: (H, W, 3)
    # goal_img: (H, W, 3)
    # return: float
    # compute the difference between the current and goal images
    assert current_img.size == goal_img.size
    # assert current_img.size[2] == 3
    
    diff = np.abs(np.array(current_img) - np.array(goal_img))
    print('mean diff: ', np.mean(diff))
    return np.mean(diff)

def get_reward_v2(current_img, goal_img):
    # current_img: (H, W, 3)
    # goal_img: (H, W, 3)
    # return: float
    # compute the percentage of pixels that differ significantly between current and goal images
    assert current_img.size == goal_img.size
    
    # Convert images to numpy arrays and compute absolute difference
    diff = np.abs(np.array(current_img) - np.array(goal_img))
    
    # Define a threshold for what constitutes a "large" difference
    # Using 50 as threshold (out of 255 possible values)
    threshold = 5
    
    # Count pixels where any RGB channel exceeds threshold
    large_diff_pixels = np.sum(np.any(diff > threshold, axis=2))
    total_pixels = diff.shape[0] * diff.shape[1]
    
    # Calculate percentage of pixels with large differences
    percentage_diff = (large_diff_pixels / total_pixels) * 100
    
    print('Percentage of pixels with large differences: ', percentage_diff)
    return percentage_diff


def get_goal_img(env_for_goal, start_t, episode, num_steps_ahead=10):
    # Reset the goal environment to the state at start_t
    # minari environments don't have a direct set_state.
    # We need to simulate steps from the beginning of the episode or reset to a known state
    # and then step to the desired point.
    
    # For minari environments, the most reliable way to get to a specific state
    # is to reset and then step through the episode actions.
    env_for_goal.reset() # Reset to the start of the episode
    
    # Step the goal environment to the 'start_t' point of the episode
    # This prepares the environment to then step 'num_steps_ahead'
    for i in range(start_t):
        env_for_goal.step(episode.actions[i])
        
    # Now, step the goal environment forward for the 'num_steps_ahead' steps
    for i in range(num_steps_ahead):
        current_step_in_episode = start_t + i
        if current_step_in_episode < len(episode.actions): # Ensure we don't go out of bounds
            print(f"Goal Env: Stepping to episode action {current_step_in_episode}")
            obs, reward, terminated, truncated, info = env_for_goal.step(episode.actions[current_step_in_episode])
            if terminated or truncated:
                print(f"Goal Env: Episode terminated or truncated early at step {current_step_in_episode}")
                break
        else:
            print(f"Goal Env: Reached end of episode for goal image at step {current_step_in_episode}")
            break

    goal_img = Image.fromarray(env_for_goal.render())
    
    return goal_img
