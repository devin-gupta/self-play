# Kitchen Environment Dataset

This module provides a PyTorch dataset for loading and processing frame pairs from the D4RL kitchen environment datasets.

## Features

- Loads data from multiple D4RL kitchen datasets:
  - `D4RL/kitchen/mixed-v2`
  - `D4RL/kitchen/complete-v2`
  - `D4RL/kitchen/partial-v2`
- Generates pairs of frames with configurable future step lookahead (k)
- Automatically downloads and caches datasets
- Supports dataset rebuilding with `force_rebuild=True`
- Returns normalized tensors in CHW format

## Usage

```python
from src.data.dataloader import KitchenPairDataset

# Load or create dataset with k=1 (default)
dataset = KitchenPairDataset(k_step_future=1)

# Load with different k values
dataset_k3 = KitchenPairDataset(k_step_future=3)
dataset_k5 = KitchenPairDataset(k_step_future=5)

# Force rebuild dataset
dataset = KitchenPairDataset(k_step_future=1, force_rebuild=True)
```

## Dataset Structure

Each item in the dataset returns a tuple of:
- `current_frame`: Tensor of shape [3, H, W] normalized to [-1, 1]
- `future_frame`: Tensor of shape [3, H, W] normalized to [-1, 1]
- `dataset`: Name of the source dataset
- `episode_num`: Episode number
- `current_step`: Current timestep

## Data Processing

1. For each dataset:
   - Loads 5 episodes
   - Processes each episode to collect frames and transitions
   - Truncates at the last task transition
2. Creates frame pairs with k-step lookahead
3. Saves processed pairs to pickle files for faster loading
4. Files are stored in `data/kitchen_pairs/pairs_k{k}.pkl`

## Example

```python
# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate over batches
for current_frames, future_frames, datasets, episodes, current_steps in dataloader:
    # current_frames: [B, 3, H, W]
    # future_frames: [B, 3, H, W]
    # datasets: List of dataset names
    # episodes: List of episode numbers
    # current_steps: List of timesteps
    ...
```

## Data Format

- **Input**: RGB images from the kitchen environment (480x480)
- **Output**: Pairs of frames (current, future)
- **Normalization**: Images are normalized to [-1,1] range
- **Format**: PyTorch tensors in CHW format (Channels, Height, Width)

## File Structure

```
data/
  kitchen_pairs/
    pairs_k1.pkl    # Dataset with k=1
    pairs_k3.pkl    # Dataset with k=3
    pairs_k5.pkl    # Dataset with k=5
```

## Implementation Details

- **Progress Tracking**: Uses tqdm to show progress during:
  - Dataset processing
  - Episode processing within each dataset
  - Frame pair creation
- **Episode Processing**:
  - Each episode is truncated at the last task completion
  - Frames are collected until either:
    - All 4 tasks are completed
    - Episode terminates
    - Episode is truncated
- **Frame Pairs**:
  - For each frame t, creates a pair with frame t+k
  - Uses the last frame for padding when t+k exceeds episode length
  - Ensures consistent pair generation across different k values

## Notes

- The dataset automatically handles padding at episode boundaries
- Each episode is truncated at the last task completion
- The number of pairs decreases slightly with larger k values
- Images are stored as uint8 arrays and converted to normalized tensors on-the-fly
- Progress bars show:
  - Percentage complete
  - Number of items processed
  - Processing speed
  - Estimated time remaining 