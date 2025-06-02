import minari
import numpy as np
import os
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring disabled.")
    print("Install with: pip install psutil")

class KitchenPairDataset(Dataset):
    def __init__(self, k_step_future=1, data_dir="data/kitchen_pairs", force_rebuild=False, 
                 max_episodes_per_batch=5, max_frames_in_memory=200, target_size=128):
        """
        Args:
            k_step_future: Number of steps to look ahead for the next frame
            data_dir: Directory to store/load the dataset
            force_rebuild: If True, rebuild the dataset even if it exists
            max_episodes_per_batch: Maximum number of episodes to process at once
            max_frames_in_memory: Maximum number of frames to keep in memory
            target_size: Size to resize images to (default: 128x128)
        """
        self.k = k_step_future
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        # Include target size in filename
        self.pairs_file = self.data_dir / f"pairs_k{self.k}_{target_size}x{target_size}.pkl"
        self.max_episodes_per_batch = max_episodes_per_batch
        self.max_frames_in_memory = max_frames_in_memory
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate or load pairs
        if not self.pairs_file.exists() or force_rebuild:
            self.pairs = self._generate_pairs()
            self._save_pairs()
        else:
            self.pairs = self._load_pairs()
            
        print(f"Loaded {len(self.pairs)} pairs with k={k_step_future}")
        print(f"Images pre-resized to {target_size}x{target_size}")
        
        # Convert to tensors for faster access
        self._convert_to_tensors()
    
    def _convert_to_tensors(self):
        """Convert all pairs to pre-processed tensors for faster training."""
        print("Converting to tensors and pre-processing images...")
        converted_pairs = []
        
        # Check if MPS is available for potential optimizations
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        for i, (current_frame, future_frame, dataset, episode_num, current_step) in enumerate(tqdm(self.pairs, desc="Converting to tensors")):
            # Convert numpy arrays to tensors and normalize (fix negative stride issue)
            current_tensor = torch.from_numpy(current_frame.copy()).float()
            future_tensor = torch.from_numpy(future_frame.copy()).float()
            
            # Normalize to [-1, 1] and convert to CHW format
            current_tensor = (current_tensor / 127.5 - 1).permute(2, 0, 1)  # [3, H, W]
            future_tensor = (future_tensor / 127.5 - 1).permute(2, 0, 1)    # [3, H, W]
            
            # Keep tensors on CPU for storage efficiency, move to device during training
            converted_pairs.append((current_tensor, future_tensor, dataset, episode_num, current_step))
        
        self.pairs = converted_pairs
        print(f"Tensor conversion complete! (Device: {device})")
    
    def _generate_pairs(self):
        """Generate frame pairs from all kitchen environment datasets."""
        # List of datasets to process
        dataset_names = [
            # 'D4RL/kitchen/mixed-v2',
            'D4RL/kitchen/complete-v2',
            # 'D4RL/kitchen/partial-v2'
        ]
        
        all_pairs = []
        
        # Process each dataset
        for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
            print(f"\nProcessing dataset: {dataset_name}")
            try:
                dataset = minari.load_dataset(dataset_name, download=True)
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                continue
                
            dataset.set_seed(seed=2)
            env = dataset.recover_environment(render_mode='rgb_array')
            env.reset(seed=2)
            
            total_episodes = len(dataset)
            print(f"Total episodes in dataset: {total_episodes}")
            
            # Process episodes in batches
            for batch_start in range(0, total_episodes, self.max_episodes_per_batch):
                batch_end = min(batch_start + self.max_episodes_per_batch, total_episodes)
                episodes = dataset.sample_episodes(n_episodes=batch_end-batch_start)
                
                batch_pairs = []
                for idx, episode in enumerate(tqdm(episodes, desc=f"Processing episodes {batch_start}-{batch_end}", leave=False)):
                    env.reset()
                    frames = []
                    transitions = []
                    seen_tasks = set()
                    episode_num = batch_start + idx
                    frame_offset = 0
                    
                    # Process episode
                    for t, action in enumerate(episode.actions):
                        if len(frames) >= self.max_frames_in_memory:
                            # Process and clear frames if we hit the memory limit
                            self._process_frames_to_pairs(frames, transitions, batch_pairs, 
                                                        dataset_name, episode_num, frame_offset)
                            frame_offset += len(frames)  # Update offset for next batch
                            frames = []
                            transitions = []
                            # Force garbage collection more frequently
                            gc.collect()
                            
                        frame = env.render()
                        frames.append(frame)
                        
                        obs, reward, terminated, truncated, info = env.step(action)
                        for task in info.get('step_task_completions', []):
                            if task not in seen_tasks:
                                seen_tasks.add(task)
                                transitions.append((task, t))
                        
                        if len(info.get('episode_task_completions', [])) == 4 or terminated or truncated:
                            break
                    
                    # Process remaining frames
                    if frames:
                        self._process_frames_to_pairs(frames, transitions, batch_pairs, 
                                                    dataset_name, episode_num, frame_offset)
                        # Clear frames immediately after processing
                        frames.clear()
                    
                    if (idx + 1) % 5 == 0:  # More frequent reporting
                        print(f"\n[Dataset: {dataset_name}, Episode {batch_start + idx + 1}]")
                        print(f"Transitions = {transitions}")
                        print(f"Batch pairs so far: {len(batch_pairs)}")
                        
                        # Memory monitoring
                        if PSUTIL_AVAILABLE:
                            process = psutil.Process()
                            memory_mb = process.memory_info().rss / 1024 / 1024
                            print(f"Current memory usage: {memory_mb:.1f} MB")
                
                # Add batch pairs to all pairs
                all_pairs.extend(batch_pairs)
                print(f"Added {len(batch_pairs)} pairs from batch {batch_start}-{batch_end}")
                
                # Aggressive memory cleanup
                del batch_pairs
                del episodes
                gc.collect()
            
            env.close()
        
        print(f"\nTotal pairs created: {len(all_pairs)}")
        return all_pairs
    
    def _process_frames_to_pairs(self, frames, transitions, pairs, dataset_name, episode_num, frame_offset=0):
        """Process frames into pairs and add to the pairs list."""
        if not frames:
            return
            
        # Truncate at last transition if any
        if transitions:
            last_step = transitions[-1][1]
            cut = last_step + 1
            frames = frames[:cut]
        
        # Resize all frames at once for efficiency
        resized_frames = self._resize_frames_batch(frames)
        
        # Create pairs for every frame
        T = len(resized_frames)
        for t in range(T):
            t2 = min(t + self.k, T - 1)
            current_step = frame_offset + t  # Track absolute timestep in episode
            pairs.append((resized_frames[t], resized_frames[t2], dataset_name, episode_num, current_step))
    
    def _resize_frames_batch(self, frames):
        """Resize a batch of frames efficiently."""
        if not frames or frames[0].shape[:2] == (self.target_size, self.target_size):
            return frames
            
        # Convert to tensor batch for efficient resizing
        frames_tensor = torch.from_numpy(np.stack(frames)).float()  # [N, H, W, 3]
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [N, 3, H, W]
        
        # Resize batch
        resized_tensor = F.interpolate(
            frames_tensor, 
            size=(self.target_size, self.target_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert back to numpy
        resized_tensor = resized_tensor.permute(0, 2, 3, 1)  # [N, H, W, 3]
        resized_frames = [frame.numpy().astype(np.uint8) for frame in resized_tensor]
        
        return resized_frames
    
    def _save_pairs(self):
        """Save the generated pairs to a pickle file."""
        with open(self.pairs_file, 'wb') as f:
            pickle.dump(self.pairs, f)
        print(f"Saved {len(self.pairs)} pairs to {self.pairs_file}")
    
    def _load_pairs(self):
        """Load pairs from the pickle file."""
        with open(self.pairs_file, 'rb') as f:
            pairs = pickle.load(f)
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Return a pair of frames (current, future) with metadata."""
        # Frames are already pre-processed tensors - just return them!
        current_frame, future_frame, dataset, episode_num, current_step = self.pairs[idx]
        return current_frame, future_frame, dataset, episode_num, current_step

# Note: This is the optimized version of KitchenPairDataset

# Example usage:
if __name__ == "__main__":
    # 1. Try loading existing datasets first
    print("\n=== Loading/Creating datasets ===")
    for k in [
        # 1,
        # 3,
        5,
    ]:
        print(f"\nTrying k={k}:")
        try:
            # Try to load existing dataset first with memory-efficient settings
            dataset = KitchenPairDataset(
                k_step_future=k, 
                force_rebuild=False,
                max_episodes_per_batch=3,  # Process even fewer episodes at once
                max_frames_in_memory=100   # Keep even fewer frames in memory
            )
            print(f"Successfully loaded existing dataset with {len(dataset)} pairs")
        except FileNotFoundError:
            # If dataset doesn't exist, create it
            print(f"No existing dataset found for k={k}, creating new one...")
            dataset = KitchenPairDataset(
                k_step_future=k, 
                force_rebuild=True,
                max_episodes_per_batch=3,
                max_frames_in_memory=100
            )
        
        # Test the dataset with a memory-efficient DataLoader
        if k == 5:  # Only test with k=5
            print("\n=== Testing DataLoader batching ===")
            print(f"\nTotal number of pairs: {len(dataset.pairs)}")
            print(f"Each pair contains 2 frames (current, future)")
            
            # Create a memory-efficient DataLoader
            dataloader = DataLoader(
                dataset, 
                batch_size=2,              # Smaller batch size
                shuffle=True,
                num_workers=0,             # Single-threaded to avoid macOS multiprocessing issues
                pin_memory=False,          # Disabled for MPS
                persistent_workers=False,  # Disabled for single-threaded
                prefetch_factor=None       # Not needed for single-threaded
            )
            
            for batch_idx, (current_frames, future_frames, datasets, episode_nums, timesteps) in enumerate(dataloader):
                print(f"\nBatch {batch_idx}:")
                print(f"Current frames shape: {current_frames.shape}")
                print(f"Future frames shape: {future_frames.shape}")
                print(f"Value range: [{current_frames.min():.2f}, {current_frames.max():.2f}]")
                print(f"Datasets: {datasets}")
                print(f"Episode numbers: {episode_nums}")
                print(f"Timesteps: {timesteps}")
                
                # Only show first batch
                if batch_idx == 0:
                    break