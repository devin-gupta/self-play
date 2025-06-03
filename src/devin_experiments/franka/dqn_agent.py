import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np # Import numpy for converting to tensor
from qnet import QNetwork, ReplayBuffer

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, obs_size, action_size, learning_rate, gamma,
                 epsilon_start, epsilon_end, epsilon_decay,
                 replay_buffer_capacity, batch_size, target_update_freq,
                 discrete_actions, device): # Added 'device' parameter
        self.device = device # Store the device

        self.q_network = QNetwork(obs_size, action_size, device).to(device) # Pass device to QNetwork and move to device
        self.target_q_network = QNetwork(obs_size, action_size, device).to(device) # Pass device to QNetwork and move to device
        self.target_q_network.load_state_dict(self.q_network.state_dict()) # Initialize target network with Q-network weights
        self.target_q_network.eval() # Set target network to evaluation mode (no gradients)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss() # Mean Squared Error loss for Q-value prediction
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

        # Store hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.num_discrete_actions = action_size # Renamed for clarity
        self.discrete_actions = discrete_actions # Store discrete actions

    def choose_action(self, observation):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: choose a random discrete action index
            return random.randrange(self.num_discrete_actions)
        else:
            # Exploit: choose the action with the highest predicted Q-value
            with torch.no_grad(): # Disable gradient calculation for inference
                # Convert observation to tensor, add a batch dimension, and move to the correct device
                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
                self.q_network.eval() # Set Q-network to evaluation mode
                q_values = self.q_network(obs_tensor)
                self.q_network.train() # Set Q-network back to training mode
                # Return the index of the action with the maximum Q-value (move back to CPU for .item())
                return q_values.argmax().item()

    def learn(self):
        # Only learn if there are enough experiences in the replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors and move to the correct device
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device) # Use uint8 then float for consistency

        # Compute Q-values for current states using the Q-network
        current_q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values for next states using the target Q-network
        # Use .detach() to ensure no gradients are computed for the target network
        next_q_values = self.target_q_network(next_states).max(1)[0].detach()

        # Compute the target for the Q-value update: R + gamma * max_a' Q_target(s', a')
        target_q_values = rewards + self.gamma * next_q_values.unsqueeze(1) * (1 - dones)

        # Compute the loss between current Q-values and target Q-values
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize the Q-network
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward()             # Backpropagate the loss
        self.optimizer.step()      # Update network weights

    def update_target_network(self):
        # Copy weights from Q-network to target Q-network
        self.target_q_network.load_state_dict(self.q_network.state_dict())