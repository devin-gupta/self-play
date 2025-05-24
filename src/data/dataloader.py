import minari
import numpy as np
import os
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

class KitchenPairDataset(Dataset):
    def __init__(self, k_step_future=1, data_dir="data/kitchen_pairs", force_rebuild=False):
        """
        Args:
            k_step_future: Number of steps to look ahead for the next frame
            data_dir: Directory to store/load the dataset
            force_rebuild: If True, rebuild the dataset even if it exists
        """
        self.k = k_step_future
        self.data_dir = Path(data_dir)
        self.pairs_file = self.data_dir / f"pairs_k{self.k}.pkl"
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate or load pairs
        if not self.pairs_file.exists() or force_rebuild:
            self.pairs = self._generate_pairs()
            self._save_pairs()
        else:
            self.pairs = self._load_pairs()
            
        print(f"Loaded {len(self.pairs)} pairs with k={k_step_future}")
    
    def _generate_pairs(self):
        """Generate frame pairs from all kitchen environment datasets."""
        # List of datasets to process
        dataset_names = [
            # 'D4RL/kitchen/mixed-v2',
            'D4RL/kitchen/complete-v2',
            # 'D4RL/kitchen/partial-v2'
        ]
        
        all_truncated_episodes = []
        
        # Process each dataset
        for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
            print(f"\nProcessing dataset: {dataset_name}")
            try:
                # Try to load dataset, download if not found
                dataset = minari.load_dataset(dataset_name, download=True)
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                print("Skipping this dataset...")
                continue
                
            dataset.set_seed(seed=2)
            env = dataset.recover_environment(render_mode='rgb_array')
            env.reset(seed=2)  # Add fixed seed for environment
            
            # Get total number of episodes in dataset
            total_episodes = len(dataset)
            print(f"Total episodes in dataset: {total_episodes}")
            
            # Get all episodes by sampling without replacement
            episodes = dataset.sample_episodes(n_episodes=5)
            
            for idx, episode in enumerate(tqdm(episodes, desc=f"Processing {dataset_name} episodes", leave=False), start=1):
                env.reset()
                frames = []            # store every rendered frame
                transitions = []       # (task, step) in order
                seen_tasks = set()
                
                # Roll through actions, collecting frames + transitions
                for t, action in enumerate(episode.actions):
                    frame = env.render()          # H×W×3 uint8 array
                    frames.append(frame)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    for task in info.get('step_task_completions', []):
                        if task not in seen_tasks:
                            seen_tasks.add(task)
                            transitions.append((task, t))
                    
                    if len(info.get('episode_task_completions', [])) == 4 or terminated or truncated:
                        break
                
                # Truncate at last transition
                if transitions:
                    last_step = transitions[-1][1]
                    cut = last_step + 1
                    
                    all_truncated_episodes.append({
                        'frames': frames[:cut],
                        'actions': episode.actions[:cut],
                        'rewards': episode.rewards[:cut],
                        'transitions': transitions,
                        'last_step': last_step,
                        'dataset': dataset_name  # Keep track of which dataset this came from
                    })
                    if idx % 10 == 0:  # Print details every 10 episodes
                        print(f"\n[Dataset: {dataset_name}, Episode {idx}]")
                        print(f"Transitions = {transitions}")
                        print(f"Number of frames collected: {len(frames)}")
                        print(f"Number of frames after truncation: {len(frames[:cut])}")
                else:
                    if idx % 10 == 0:  # Print details every 10 episodes
                        print(f"[Dataset: {dataset_name}, Episode {idx}] No tasks completed")
            
            env.close()
        
        # Build pairs across all episodes from all datasets
        pairs = []
        for ep_idx, ep in enumerate(tqdm(all_truncated_episodes, desc="Creating frame pairs"), 1):
            F = ep['frames']                 # length = T
            T = len(F)
            if ep_idx % 10 == 0:  # Print details every 10 episodes
                print(f"\n[Dataset: {ep['dataset']}, Episode {ep_idx}] Creating pairs:")
                print(f"Total frames: {T}")
                print(f"Last transition step: {ep['last_step']}")
            
            # Create pairs for every frame in the episode
            for t in range(T):
                # Calculate future frame index, pad with last frame if needed
                t2 = min(t + self.k, T - 1)  # This ensures we use the last frame for padding
                pairs.append((
                    F[t],           # current frame
                    F[t2],          # future frame
                    ep['dataset'],  # dataset name
                    ep_idx,         # episode number
                    t               # current timestep
                ))
        
        print(f"\nTotal episodes processed: {len(all_truncated_episodes)}")
        print(f"Total pairs created: {len(pairs)}")
        return pairs
    
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
            # Try to load existing dataset first
            dataset = KitchenPairDataset(k_step_future=k, force_rebuild=False)
            print(f"Successfully loaded existing dataset with {len(dataset.pairs)} pairs")
                
        except FileNotFoundError:
            # If dataset doesn't exist, create it
            print(f"No existing dataset found for k={k}, creating new one...")
            dataset = KitchenPairDataset(k_step_future=k, force_rebuild=True)