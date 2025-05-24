# Kitchen Pair Dataset

This dataloader creates and manages pairs of frames from the D4RL kitchen environment for training frame prediction models.

## Overview

The `KitchenPairDataset` class:
- Loads episodes from three D4RL kitchen datasets (mixed, complete, and partial)
- Tracks task completions and transitions
- Creates pairs of frames (current frame, future frame) with configurable look-ahead steps (k=1,3,5)
- Saves/loads the processed data to avoid recomputing
- Provides progress tracking during dataset generation

## Dataset Processing

The dataloader processes data in three stages:
1. **Dataset Loading**: Loads episodes from each D4RL kitchen dataset
2. **Episode Processing**: For each episode:
   - Tracks task completions and transitions
   - Collects frames and actions
   - Truncates at the last successful task completion
3. **Pair Generation**: Creates frame pairs with configurable look-ahead steps

## Usage

```python
from src.data.dataloader import KitchenPairDataset
from torch.utils.data import DataLoader

# Basic usage - load existing dataset or create new one
dataset = KitchenPairDataset(
    k_step_future=1,              # How many steps ahead to predict (1, 3, or 5)
    data_dir="data/kitchen_pairs", # Where to save/load the data
    force_rebuild=False           # Set to True to regenerate the dataset
)

# Create dataloader for training
dataloader = DataLoader(
    dataset,
    batch_size=32,    # Adjust based on your GPU memory
    shuffle=True,     # Shuffle for training
    num_workers=4     # Parallel data loading
)

# Use in training loop
for current_frames, future_frames in dataloader:
    # current_frames: [B, 3, H, W] tensor in [-1,1] range
    # future_frames: [B, 3, H, W] tensor in [-1,1] range
    # B = batch_size, H = 480, W = 480
    ...

# Example: Load multiple k values
datasets = {}
for k in [1, 3, 5]:
    datasets[k] = KitchenPairDataset(
        k_step_future=k,
        force_rebuild=False  # Will load existing if available
    )
    print(f"Loaded {len(datasets[k])} pairs for k={k}")

# Example: Force rebuild a specific k value
dataset_k3 = KitchenPairDataset(
    k_step_future=3,
    force_rebuild=True  # Will regenerate the dataset
)

# Example: Access raw pairs (before tensor conversion)
current_frame, future_frame = dataset.pairs[0]
print(f"Raw frame shapes: {current_frame.shape}, {future_frame.shape}")  # (480, 480, 3)
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