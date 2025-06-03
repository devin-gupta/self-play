import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

# --- Q-Network Definition ---
class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size, device): # Add 'device' parameter
        super(QNetwork, self).__init__()
        # Define a simple feed-forward neural network
        self.fc1 = nn.Linear(obs_size, 128) # First fully connected layer
        self.relu1 = nn.ReLU()             # ReLU activation function
        self.fc2 = nn.Linear(128, 128)     # Second fully connected layer
        self.relu2 = nn.ReLU()             # ReLU activation function
        self.fc3 = nn.Linear(128, action_size) # Output layer, size equals number of discrete actions
        self.to(device) # Move the entire network to the specified device

    def forward(self, x):
        # Forward pass through the network
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x) # Output Q-values for each action

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity, device): # Add 'device' parameter
        # Initialize a deque (double-ended queue) with a maximum capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device # Store the device

    def push(self, state, action, reward, next_state, done):
        # Add a new experience tuple to the buffer.
        # Store as NumPy arrays to avoid immediate tensor conversion and device transfer,
        # which can be memory inefficient if the buffer is very large.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)
        # Unpack the batch into separate lists of states, actions, etc.
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors and move them to the specified device
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.long).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            # Dones should be float for multiplication with gamma * next_q_values
            torch.tensor(dones, dtype=torch.float32).to(self.device)
        )

    def __len__(self):
        # Return the current size of the buffer
        return len(self.buffer)