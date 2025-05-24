import minari
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time # For pauses between steps

# --- 1. Load the Trained Model ---
print("Loading the trained Behavior Cloning model 'bc_kitchen_agent.keras'...")
try:
    model = keras.models.load_model('bc_kitchen_agent.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'bc_kitchen_agent.keras' exists in the same directory.")
    exit()

# --- 2. Initialize the Minari Environment for Rendering ---
print("Initializing Minari environment for rendering...")
# We need to recover the environment type to create a new instance for rendering
# One way to do this is to load a dummy dataset just to get the env_name
dataset = minari.load_dataset('D4RL/kitchen/mixed-v2')
env = dataset.recover_environment(render_mode='human')
# dataset.close() # Close the dummy dataset

# Get state and action dimensions from the environment for correct input shaping
state_dim = env.observation_space['observation'].shape[0]
action_dim = env.action_space.shape[0]
print(f"Observation space dimension: {state_dim}")
print(f"Action space dimension: {action_dim}")


# --- 3. Run and Render Evaluation Episodes ---
num_render_episodes = 3# Number of episodes to render
max_steps_per_episode = 100 # Max steps to visualize per episode (can be longer than training eval)
render_delay = 0.05 # Small delay to make rendering visible

print(f"\nStarting rendering for {num_render_episodes} episodes, {max_steps_per_episode} steps per episode...")

for i in range(num_render_episodes):
    obs, info = env.reset()
    total_reward = 0
    print(f"\n--- Rendering Episode {i+1}/{num_render_episodes} ---")

    for t in range(max_steps_per_episode):
        # Ensure the observation is in the correct shape (batch_size, state_dim)
        obs_reshaped = obs['observation']
        obs_input_to_model = np.expand_dims(obs_reshaped, axis=0) # Make it (1, state_dim)

        # Predict action using the loaded BC model
        predicted_action = model.predict(obs_input_to_model, verbose=0)[0] # [0] to remove batch dim

        # Take the predicted action in the environment
        obs, reward, terminated, truncated, info = env.step(predicted_action)
        total_reward += reward

        # Render the environment
        env.render()
        time.sleep(render_delay) # Pause for a short duration

        if terminated or truncated:
            print(f"Episode terminated/truncated at step {t+1}. Total reward: {total_reward:.2f}")
            break
    else: # This else block executes if the loop completes without a 'break'
        print(f"Episode finished max steps ({max_steps_per_episode}). Total reward: {total_reward:.2f}")

env.close() # Close the environment after rendering
print("\nRendering complete.")