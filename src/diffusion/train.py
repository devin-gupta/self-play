"""
train.py
"""
import math
import minari
import numpy as np
from pympler import asizeof
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .model import *

# --- 1. Diffusion Hyperparameters & Scheduler ---
TIMESTEPS = 1000  # Number of diffusion timesteps

def get_linear_beta_schedule(timesteps):
    """
    Returns a linear beta schedule from beta_start to beta_end.
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

betas = get_linear_beta_schedule(TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0) # alpha_bar
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # alpha_bar_{t-1}

# Precompute values needed for noising (Equation 4 in DDPM paper)
# x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def add_noise_to_images(original_images, t, device):
    """Adds noise to images x_0 to get x_t."""
    # original_images: [B, C, H, W]
    # t: [B] tensor of timesteps
    batch_size = original_images.shape[0]
    noise = torch.randn_like(original_images, device=device) # epsilon

    # Gather coefficients for each t in the batch
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device).view(batch_size, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device).view(batch_size, 1, 1, 1)

    noisy_images = sqrt_alphas_cumprod_t * original_images + \
                   sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_images, noise # Return noisy image and the noise that was added (target)

# --- 2. Dataset ---
class KitchenDataset(Dataset):
    def __init__(self, k_step_future=1):
        complete_dataset = minari.load_dataset('D4RL/kitchen/complete-v2')
        mixed_dataset = minari.load_dataset('D4RL/kitchen/mixed-v2')
        partial_dataset = minari.load_dataset('D4RL/kitchen/partial-v2')
        self.dataset = [complete_dataset + mixed_dataset + partial_dataset]
        self.current_dataset_idx = 0
        self.num_samples = len(self.dataset[0])

    def __len__(self):
        return self.num_samples
    
    def _get_
    
    def __getitem__(self, idx):
        

# --- 3. Training Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_RESOLUTION = 480 # Should match your data and model's initial_img_resolution
BATCH_SIZE = 4 # Adjust based on your GPU memory
LEARNING_RATE = 1e-4 # A common starting point
EPOCHS = 100 # Number of training epochs

# --- 4. Model, Optimizer, Loss ---
model = ConditionalUNet(
    in_img_channels=6,
    out_img_channels=3,
    base_channels=64,       # Tunable
    channel_mults=(1, 2, 3, 4), # Tunable
    num_res_blocks_per_level=2,
    attention_resolutions=(IMG_RESOLUTION // 4, IMG_RESOLUTION // 8), # e.g., 120, 60 for 480px
    time_emb_dim=256,
    time_emb_mlp_dim=1024,
    initial_img_resolution=IMG_RESOLUTION
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss() # Loss between predicted noise and actual added noise

# --- 5. Training Loop ---
if __name__ == "__main__": # Guard for multiprocessing in DataLoader
    print(f"Using device: {DEVICE}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in U-Net: {num_params:,}")

    # Create dataset and dataloader
    # IMPORTANT: Replace DummyKitchenDataset with your actual dataset pipeline
    # Ensure your images are preprocessed (e.g., resized, normalized to [-1, 1])
    train_dataset = KitchenDataset(num_samples=128, img_res=IMG_RESOLUTION) # Small for demo
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (current_img_batch, next_img_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Move data to device
            current_img_batch = current_img_batch.to(DEVICE) # [B, 3, H, W]
            next_img_batch = next_img_batch.to(DEVICE)       # [B, 3, H, W] (this is x_0 for the next state)

            # Sample random timesteps t for each image in the batch
            t = torch.randint(0, TIMESTEPS, (current_img_batch.shape[0],), device=DEVICE).long()

            # Add noise to next_img_batch (x_0) to get x_t, and get the added noise (epsilon)
            noisy_next_img, added_noise_epsilon = add_noise_to_images(next_img_batch, t, DEVICE)

            # Prepare model input: concatenate noisy_next_image and current_image
            model_input = torch.cat([noisy_next_img, current_img_batch], dim=1) # [B, 6, H, W]

            # Predict noise
            predicted_noise_epsilon = model(model_input, t)

            # Calculate loss
            loss = criterion(predicted_noise_epsilon, added_noise_epsilon)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 20 == 0: # Log every 20 batches
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        avg_epoch_loss = total_loss / len(train_dataloader)
        print(f"--- Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_epoch_loss:.4f} ---")

        # Optional: Save model checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"unet_kitchen_epoch_{epoch+1}.pth")
            print(f"Saved model checkpoint at epoch {epoch+1}")

    print("Training complete.")
