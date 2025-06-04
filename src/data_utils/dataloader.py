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
import argparse

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring disabled.")
    print("Install with: pip install psutil")

class KitchenPairDataset(Dataset):
    def __init__(self, k_step_future=10, data_dir="data/kitchen_pairs", force_rebuild=False, 
                 max_episodes_per_batch=5, max_frames_in_memory=200, target_size=256):
        """
        Args:
            k_step_future: Number of steps to look ahead for the next frame (default: 10)
            data_dir: Directory to store/load the dataset
            force_rebuild: If True, rebuild the dataset even if it exists
            max_episodes_per_batch: Maximum number of episodes to process at once
            max_frames_in_memory: Maximum number of frames to keep in memory
            target_size: Size to resize images to (default: 256x256)
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
            'D4RL/kitchen/complete-v2',
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
                    
                    if (idx + 1) % 20 == 0:  # Reduced frequency of reporting
                        print(f"\n[Dataset: {dataset_name}, Episode {batch_start + idx + 1} processed in current batch]")
                        
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
    
    # --- Add argument parsing ---
    parser = argparse.ArgumentParser(description="Load or generate Kitchen dataset pairs.")
    parser.add_argument(
        "--k", 
        type=int, 
        default=10,  # Default k value
        help="Number of steps to look ahead for the future frame (k_step_future)."
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuild the dataset even if a pre-existing one is found."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/kitchen_pairs",
        help="Directory to store/load the dataset."
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Target size to resize images to (e.g., 256 for 256x256)."
    )

    args = parser.parse_args()
    # --- End argument parsing ---

    # Use the parsed k value
    k_value = args.k
    print(f"\nProcessing for k={k_value}:")
    
    try:
        # Try to load existing dataset first with memory-efficient settings
        dataset = KitchenPairDataset(
            k_step_future=k_value,
            data_dir=args.data_dir,
            force_rebuild=args.force_rebuild,
            max_episodes_per_batch=2, # Reduce for memory efficiency if needed
            max_frames_in_memory=100, # Reduce for memory efficiency if needed
            target_size=args.target_size 
        )
        
        print(f"\n--- Dataset for k={k_value} (size: {args.target_size}x{args.target_size}) ---")
        print(f"Total pairs: {len(dataset)}")
        
        # Optional: Get a sample and print its shape
        if len(dataset) > 0:
            current_frame, future_frame, _, _, _ = dataset[0]
            print(f"Sample current_frame shape: {current_frame.shape}")
            print(f"Sample future_frame shape: {future_frame.shape}")

            # Verify tensor properties (optional)
            print(f"Current frame dtype: {current_frame.dtype}, min: {current_frame.min():.2f}, max: {current_frame.max():.2f}")
            print(f"Future frame dtype: {future_frame.dtype}, min: {future_frame.min():.2f}, max: {future_frame.max():.2f}")

    except Exception as e:
        print(f"Error processing for k={k_value}: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Dataset processing complete. ===")