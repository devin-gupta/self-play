import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
from qnet import QNetwork, ReplayBuffer # Import QNetwork from q_network.py
from dqn_agent import DQNAgent # Import DQNAgent from dqn_agent.py
from custom_franka_env import CustomFrankaEnv # Import the custom wrapper for FrankaKitchen-v1

# Register the environment
gym.register_envs(gymnasium_robotics)

# --- Device Setup ---
# Check if MPS is available and set the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for training.")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU for training.")


# --- Hyperparameters ---
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995 # Decay rate for epsilon
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100 # How often to update the target network
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 100

# --- Environment Setup ---
# The FrankaKitchen-v1 environment has a continuous action space (Box(9,)).
# For a simple Q-network, we need to discretize this.
# We'll define a small set of discrete actions to make Q-learning feasible.
# These actions are arbitrary and chosen for demonstration;
# in a real scenario, you'd design more meaningful discrete actions.
DISCRETE_ACTIONS = [
    np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), # 0: Joint 1
    np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), # 1: Joint 1
    np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), # 2: Joint 1
    np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), # 3: Joint 1
    np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), # 4: Joint 1
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32), # 5: Joint 1
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32), # 6: Joint 1
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32),# 7: right finger join
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32), # 8: left finger joint

    np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), # 0: Joint 1
    np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), # 1: Joint 1
    np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), # 2: Joint 1
    np.array([0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), # 3: Joint 1
    np.array([0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), # 4: Joint 1
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0], dtype=np.float32), # 5: Joint 1
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0], dtype=np.float32), # 6: Joint 1
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0], dtype=np.float32),# 7: right finger join
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5], dtype=np.float32), # 8: left finger joint
]
NUM_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)

# env = gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'])
env = CustomFrankaEnv(gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle']))

# Get observation space size
observation_space_size = env.observation_space['observation'].shape[0]

# --- Training Loop ---
# Pass hyperparameters and environment details to the agent
agent = DQNAgent(
    obs_size=observation_space_size,
    action_size=NUM_DISCRETE_ACTIONS,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY,
    replay_buffer_capacity=REPLAY_BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ,
    discrete_actions=DISCRETE_ACTIONS, # Pass discrete actions to the agent
    device=device # Pass the device to the agent
)

print(f"Starting training for {NUM_EPISODES} episodes...")

for episode in range(NUM_EPISODES):
    # Reset the environment for a new episode
    observation, info = env.reset()
    # Move observation to the correct device
    observation_tensor = torch.from_numpy(observation['observation']).float().to(device)

    assert observation['observation'].size == observation_space_size, f"Observation size mismatch, expected {observation_space_size}, got {observation.size}"
    episode_reward = 0
    done = False
    truncated = False
    step_count = 0

    while not done and not truncated and step_count < MAX_STEPS_PER_EPISODE:
        # Choose an action using the agent's epsilon-greedy policy
        discrete_action_idx = agent.choose_action(observation_tensor.cpu().numpy()) # Agent expects numpy, convert back if needed

        # Map the discrete action index back to the continuous action array
        continuous_action = DISCRETE_ACTIONS[discrete_action_idx]

        # Take a step in the environment
        next_observation, reward, done, truncated, info = env.step(continuous_action)

        # Move next_observation to the correct device
        next_observation_tensor = torch.from_numpy(next_observation['observation']).float().to(device)


        # Store the experience in the replay buffer
        # Ensure all tensors pushed to buffer are on the correct device
        agent.replay_buffer.push(
            observation_tensor.cpu().numpy(), # Store numpy arrays in buffer
            discrete_action_idx,
            reward,
            next_observation_tensor.cpu().numpy(), # Store numpy arrays in buffer
            done
        )

        # Update the current observation and episode reward
        observation = next_observation
        observation_tensor = next_observation_tensor # Update the tensor version
        episode_reward += reward
        step_count += 1

        # Perform a learning step (update Q-network)
        agent.learn()

        # Update the target network periodically
        if step_count % agent.target_update_freq == 0: # Use agent's target_update_freq
            agent.update_target_network()

        # # Render the environment
        # if render == 0:  # Render every 10 episodes
        #     print('render value: ', render, ' and episode: ', episode)
        #     env.render()

    # Decay epsilon after each episode
    agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay) # Use agent's epsilon parameters

    print(f"Episode {episode + 1}/{NUM_EPISODES}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.4f}, Steps: {step_count}")

# Close the environment after training
env.close()
print("Training complete!")

env = CustomFrankaEnv(gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode='human'))

# render 5 episodes from the trained agent
for episode in range(5):
    observation, info = env.reset()
    done = False
    truncated = False
    step_count = 0

    while not done and not truncated and step_count < MAX_STEPS_PER_EPISODE:
        # Choose an action using the agent's policy
        # Ensure observation is on the correct device for the agent's Q-network
        observation_tensor = torch.from_numpy(observation['observation']).float().to(device)
        discrete_action_idx = agent.choose_action(observation_tensor.cpu().numpy()) # Agent expects numpy, convert back if needed
        continuous_action = DISCRETE_ACTIONS[discrete_action_idx]

        # Take a step in the environment
        next_observation, reward, done, truncated, info = env.step(continuous_action)

        # Update the current observation
        observation = next_observation

        # Render the environment
        env.render()

        step_count += 1

    print(f"Episode {episode + 1}/5 completed in {step_count} steps.")

env.close()
print("Visualization complete!")