import minari
import numpy as np
import os
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
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
                 max_episodes_per_batch=5, max_frames_in_memory=200):
        """
        Args:
            k_step_future: Number of steps to look ahead for the next frame
            data_dir: Directory to store/load the dataset
            force_rebuild: If True, rebuild the dataset even if it exists
            max_episodes_per_batch: Maximum number of episodes to process at once
            max_frames_in_memory: Maximum number of frames to keep in memory
        """
        self.k = k_step_future
        self.data_dir = Path(data_dir)
        self.pairs_file = self.data_dir / f"pairs_k{self.k}.pkl"
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
        print(f"Memory usage: ~{len(self.pairs) * 1.4:.1f} MB (estimated)")
    
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
        
        # Create pairs for every frame
        T = len(frames)
        for t in range(T):
            t2 = min(t + self.k, T - 1)
            current_step = frame_offset + t  # Track absolute timestep in episode
            pairs.append((frames[t], frames[t2], dataset_name, episode_num, current_step))
    
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
        current_frame, future_frame, dataset, episode_num, current_step = self.pairs[idx]
        # Make copies of the arrays to avoid negative strides
        current_frame = current_frame.copy()
        future_frame = future_frame.copy()
        # Convert to torch tensors and normalize to [-1, 1]
        current_frame = torch.from_numpy(current_frame).float() / 127.5 - 1
        future_frame = torch.from_numpy(future_frame).float() / 127.5 - 1
        # Convert from HWC to CHW format
        current_frame = current_frame.permute(2, 0, 1)
        future_frame = future_frame.permute(2, 0, 1)
        return current_frame, future_frame, dataset, episode_num, current_step

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
                force_rebuild=True,  # Force rebuild to use new format with metadata
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
                num_workers=2,             # Fewer workers
                pin_memory=True,           # Use pinned memory for faster GPU transfer
                persistent_workers=True,   # Keep workers alive between epochs
                prefetch_factor=2          # Prefetch fewer batches
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