"""
inference_demo.py - Diffusion Model Inference Demo

This script loads a trained diffusion model and demonstrates it by:
1. Loading an episode from the Minari dataset
2. For each step, using the current frame to predict the next frame
3. Displaying both the actual and predicted frames side by side
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import minari
from pathlib import Path
from tqdm import tqdm
import time

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion.model import ConditionalUNet
from diffusion.utils import add_noise_to_images

class DiffusionInference:
    def __init__(self, checkpoint_path, device='mps'):
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        self.model, self.model_config = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Diffusion parameters (should match training)
        self.timesteps = 1000
        self.img_resolution = 128  # Model expects 128x128 input
        
        print(f"Model loaded successfully!")
        print(f"Model expects {self.img_resolution}x{self.img_resolution} images")
    
    def _load_model(self, checkpoint_path):
        """Load the trained model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get model configuration from checkpoint
        model_config = checkpoint.get('model_config', {
            'in_img_channels': 6,
            'out_img_channels': 3,
            'base_channels': 32,
            'channel_mults': (1, 2, 3),
            'num_res_blocks_per_level': 1,
            'attention_resolutions': (32,),  # 128//4
            'time_emb_dim': 128,
            'time_emb_mlp_dim': 512,
            'initial_img_resolution': 128
        })
        
        # Create model
        model = ConditionalUNet(**model_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Training loss: {checkpoint.get('loss', 'unknown'):.6f}")
        
        return model, model_config
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input (resize and normalize)."""
        # Convert to tensor and resize to model resolution (fix negative stride issue)
        frame_tensor = torch.from_numpy(frame.copy()).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        frame_resized = F.interpolate(frame_tensor, size=(self.img_resolution, self.img_resolution), 
                                    mode='bilinear', align_corners=False)
        # Normalize to [-1, 1]
        frame_normalized = frame_resized / 127.5 - 1
        return frame_normalized.to(self.device)
    
    def postprocess_frame(self, tensor, target_size=480):
        """Convert model output back to displayable image."""
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
        # Resize to target display size
        if tensor.shape[-1] != target_size:
            tensor = F.interpolate(tensor, size=(target_size, target_size), 
                                 mode='bilinear', align_corners=False)
        
        # Convert to numpy and format for display
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (image * 255).astype(np.uint8)
    
    def predict_next_frame(self, current_frame, num_inference_steps=50):
        """Predict the next frame using the diffusion model."""
        with torch.no_grad():
            # Start with pure noise for the "next frame"
            noise = torch.randn(1, 3, self.img_resolution, self.img_resolution, device=self.device)
            
            # Prepare input: concatenate noise and current frame
            model_input = torch.cat([noise, current_frame], dim=1)  # [1, 6, H, W]
            
            # For simplicity, we'll use a single denoising step at a random timestep
            # In practice, you'd want to implement the full DDPM sampling process
            t = torch.randint(0, self.timesteps, (1,), device=self.device).long()
            
            # Predict the noise
            predicted_noise = self.model(model_input, t)
            
            # Simple denoising (this is a simplified approach)
            # For proper sampling, you'd implement the full DDPM reverse process
            predicted_frame = noise - predicted_noise
            
            return predicted_frame
    
    def run_episode_demo(self, dataset_name='D4RL/kitchen/complete-v2', episode_idx=0, max_steps=50):
        """Run inference demo on a specific episode."""
        print(f"\nLoading dataset: {dataset_name}")
        dataset = minari.load_dataset(dataset_name, download=True)
        dataset.set_seed(seed=42)
        
        # Get environment for rendering
        env = dataset.recover_environment(render_mode='rgb_array')
        env.reset(seed=42)
        
        # Get a specific episode
        episodes = dataset.sample_episodes(n_episodes=episode_idx+1)
        episode = episodes[episode_idx]
        
        print(f"Running inference demo on episode {episode_idx}")
        print(f"Episode length: {len(episode.actions)} steps")
        print(f"Will process first {min(max_steps, len(episode.actions))} steps")
        
        # Setup visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].set_title("Current Frame (Environment)")
        axes[1].set_title("Predicted Next Frame (Model)")
        plt.ion()  # Interactive mode for real-time updates
        
        # Reset environment and step through episode
        env.reset()
        
        for step in tqdm(range(min(max_steps, len(episode.actions))), desc="Processing steps"):
            # Get current frame from environment
            current_frame = env.render()  # [H, W, 3] numpy array
            
            # Preprocess for model
            current_frame_tensor = self.preprocess_frame(current_frame)
            
            # Predict next frame
            start_time = time.time()
            predicted_frame_tensor = self.predict_next_frame(current_frame_tensor)
            inference_time = time.time() - start_time
            
            # Postprocess for display
            predicted_frame = self.postprocess_frame(predicted_frame_tensor, target_size=480)
            
            # Display side by side
            axes[0].clear()
            axes[1].clear()
            
            axes[0].imshow(current_frame)
            axes[0].set_title(f"Current Frame - Step {step}")
            axes[0].axis('off')
            
            axes[1].imshow(predicted_frame)
            axes[1].set_title(f"Predicted Next Frame ({inference_time:.3f}s)")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.pause(0.1)  # Brief pause to see the frame
            
            # Step environment forward
            if step < len(episode.actions):
                action = episode.actions[step]
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    print(f"Episode ended at step {step}")
                    break
        
        plt.ioff()  # Turn off interactive mode
        plt.show()
        env.close()
        print("Demo completed!")

def main():
    # Configuration
    CHECKPOINT_PATH = "data/diffusion/checkpoints/model_v1/best_model.pth"  # Adjust path as needed
    DATASET_NAME = 'D4RL/kitchen/complete-v2'
    EPISODE_IDX = 0
    MAX_STEPS = 30  # Process first 30 steps
    
    try:
        # Create inference demo
        demo = DiffusionInference(CHECKPOINT_PATH)
        
        # Run the demo
        demo.run_episode_demo(
            dataset_name=DATASET_NAME,
            episode_idx=EPISODE_IDX,
            max_steps=MAX_STEPS
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have a trained model checkpoint available.")
        print("You can specify a different checkpoint path in the script.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 