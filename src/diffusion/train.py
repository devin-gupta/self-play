"""
train.py
"""
import datetime
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from tqdm import tqdm

import minari
import numpy as np
from pympler import asizeof
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


# Add the src directory to the path so we can import properly
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion.model import *
from diffusion.utils import *
from data_utils.dataloader import KitchenPairDataset

# --- Diffusion Hyperparameters & Scheduler ---
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
precomputed = {
    "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
    "sqrt_one_minus_alphas_cumprod": torch.sqrt(1. - alphas_cumprod),
}

# --- Training Configuration ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Move precomputed values to device
for key in precomputed:
    precomputed[key] = precomputed[key].to(DEVICE)

IMG_RESOLUTION = 128 # Reduced from 480 for much faster training
BATCH_SIZE = 2 # Optimized for MPS with 128x128 images
LEARNING_RATE = 1e-4 # A common starting point
EPOCHS = 100 # Number of training epochs

# Checkpointing configuration
CHECKPOINT_DIR = Path("data/diffusion/checkpoints")
CHECKPOINT_FREQ = 5  # Save checkpoint every N epochs
SAVE_BEST = True  # Save best model based on loss

# TensorBoard configuration
TENSORBOARD_LOG_DIR = Path("data/diffusion/tensorboard_logs")
LOG_FREQ = 10  # Log to tensorboard every N batches
LOG_IMAGES = True  # Log sample images and predictions

def create_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")

def create_tensorboard_dir():
    """Create TensorBoard log directory if it doesn't exist."""
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard log directory: {TENSORBOARD_LOG_DIR}")

def init_tensorboard(run_name=None):
    """Initialize TensorBoard SummaryWriter with a unique run name."""
    if run_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"diffusion_training_{timestamp}"
    
    log_dir = TENSORBOARD_LOG_DIR / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logging to: {log_dir}")
    print(f"Start TensorBoard with: tensorboard --logdir={TENSORBOARD_LOG_DIR}")
    return writer

def log_images_to_tensorboard(writer, current_img, next_img, noisy_img, predicted_noise, actual_noise, step, max_images=4):
    """Log sample images to TensorBoard for visual inspection."""
    batch_size = min(current_img.shape[0], max_images)
    
    # Denormalize images from [-1, 1] to [0, 1] for visualization
    def denormalize(img):
        return (img + 1.0) / 2.0
    
    # Log original images
    writer.add_images('1_Current_Images', denormalize(current_img[:batch_size]), step)
    writer.add_images('2_Target_Next_Images', denormalize(next_img[:batch_size]), step)
    writer.add_images('3_Noisy_Images', denormalize(noisy_img[:batch_size]), step)
    
    # Visualize noise (add 0.5 to center around gray for better visualization)
    writer.add_images('4_Predicted_Noise', predicted_noise[:batch_size] + 0.5, step)
    writer.add_images('5_Actual_Noise', actual_noise[:batch_size] + 0.5, step)
    writer.add_images('6_Noise_Difference', torch.abs(predicted_noise[:batch_size] - actual_noise[:batch_size]) + 0.5, step)

def save_checkpoint(model, optimizer, epoch, loss, loss_history, is_best=False, checkpoint_dir=CHECKPOINT_DIR):
    """Save model checkpoint with all training state."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'loss_history': loss_history,
        'model_config': {
            'in_img_channels': 6,
            'out_img_channels': 3,
            'base_channels': 32,
            'channel_mults': (1, 2, 3),
            'num_res_blocks_per_level': 1,
            'attention_resolutions': (IMG_RESOLUTION // 4,),
            'time_emb_dim': 128,
            'time_emb_mlp_dim': 512,
            'initial_img_resolution': IMG_RESOLUTION
        },
        'training_config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'timesteps': TIMESTEPS,
            'img_resolution': IMG_RESOLUTION
        }
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    # Save latest checkpoint (for easy resuming)
    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    torch.save(checkpoint, latest_path)
    
    # Save training config as JSON for easy inspection
    config_path = checkpoint_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'model_config': checkpoint['model_config'],
            'training_config': checkpoint['training_config'],
            'last_epoch': epoch,
            'last_loss': float(loss)
        }, f, indent=2)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint and restore training state."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, [], float('inf')
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss_history = checkpoint.get('loss_history', [])
    last_loss = checkpoint.get('loss', float('inf'))
    
    print(f"Resumed from epoch {epoch}, last loss: {last_loss:.6f}")
    return epoch, loss_history, last_loss

def find_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    """Find the latest checkpoint in the directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    if latest_path.exists():
        return latest_path
    
    # Fallback: find highest epoch number
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoints[-1]

# ---  Model, Optimizer, Loss ---
model = ConditionalUNet(
    in_img_channels=6,
    out_img_channels=3,
    base_channels=32,       # Reduced from 64 to save memory
    channel_mults=(1, 2, 3), # Reduced from (1, 2, 3, 4) to save memory
    num_res_blocks_per_level=1, # Reduced from 2 to save memory
    attention_resolutions=(IMG_RESOLUTION // 4,), # Reduced attention layers
    time_emb_dim=128, # Reduced from 256
    time_emb_mlp_dim=512, # Reduced from 1024
    initial_img_resolution=IMG_RESOLUTION
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss() # Loss between predicted noise and actual added noise

# Mixed precision training for faster training
scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
USE_AMP = DEVICE.type == 'cuda'  # Only use AMP on CUDA, not MPS

# --- 5. Training Loop ---
if __name__ == "__main__": # Guard for multiprocessing in DataLoader
    print(f"Using device: {DEVICE}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in U-Net: {num_params:,}")

    # Create checkpoint directory
    create_checkpoint_dir()
    
    # Create TensorBoard directory and initialize logger
    create_tensorboard_dir()
    tb_writer = init_tensorboard()

    # Try to resume from latest checkpoint
    resume_training = False
    start_epoch = 0
    loss_history = []
    best_loss = float('inf')
    global_step = 0  # For TensorBoard logging
    
    # Create dataset and dataloader first
    train_dataset = KitchenPairDataset(
        k_step_future=5,  # Use the existing k=5 dataset
        data_dir="data/kitchen_pairs",
        force_rebuild=False,  # Use existing data
        target_size=IMG_RESOLUTION  # Use 128x128 pre-processed images
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,           # Single-threaded for stability on MPS
        pin_memory=False,        # Disabled for MPS
        persistent_workers=False, # Disabled for single-threaded
        prefetch_factor=None      # Not needed for single-threaded
    )
    
    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint is not None:
        response = input(f"Found checkpoint: {latest_checkpoint.name}. Resume training? (y/n): ").lower().strip()
        if response == 'y':
            resume_training = True
            start_epoch, loss_history, last_loss = load_checkpoint(latest_checkpoint, model, optimizer)
            start_epoch += 1  # Start from next epoch
            best_loss = min(loss_history) if loss_history else float('inf')
            global_step = start_epoch * len(train_dataloader)  # Calculate global step after dataloader exists
            print(f"Resuming training from epoch {start_epoch}")

    if not resume_training:
        print("Starting training from scratch")
        
    # Log model architecture to TensorBoard
    try:
        sample_input = torch.randn(1, 6, IMG_RESOLUTION, IMG_RESOLUTION).to(DEVICE)
        sample_timestep = torch.randint(0, TIMESTEPS, (1,)).to(DEVICE)
        tb_writer.add_graph(model, (sample_input, sample_timestep))
        print("Model architecture logged to TensorBoard")
    except Exception as e:
        print(f"Could not log model graph to TensorBoard: {e}")

    print(f"Training dataset size: {len(train_dataset)} pairs")
    print(f"Starting training for {EPOCHS} epochs...")
    
    # Track training time
    training_start_time = time.time()
    print(f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Log hyperparameters to TensorBoard
    hyperparams = {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'timesteps': TIMESTEPS,
        'img_resolution': IMG_RESOLUTION,
        'base_channels': 32,
        'channel_mults': str((1, 2, 3)),
        'num_res_blocks_per_level': 1,
    }
    tb_writer.add_hparams(hyperparams, {})

    # Create epoch progress bar
    epoch_pbar = tqdm(range(start_epoch, EPOCHS), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Create batch progress bar
        batch_pbar = tqdm(train_dataloader, 
                         desc=f"Epoch {epoch+1}/{EPOCHS}", 
                         leave=False, 
                         position=1)
        
        for batch_idx, (current_img_batch, next_img_batch, datasets, episode_nums, timesteps) in enumerate(batch_pbar):
            optimizer.zero_grad()

            # Move data to device
            current_img_batch = current_img_batch.to(DEVICE) # [B, 3, H, W]
            next_img_batch = next_img_batch.to(DEVICE)       # [B, 3, H, W] (this is x_0 for the next state)

            # Sample random timesteps t for each image in the batch
            t = torch.randint(0, TIMESTEPS, (current_img_batch.shape[0],), device=DEVICE).long()

            # Add noise to next_img_batch (x_0) to get x_t, and get the added noise (epsilon)
            noisy_next_img, added_noise_epsilon = add_noise_to_images(next_img_batch, t, DEVICE, precomputed)

            # Prepare model input: concatenate noisy_next_image and current_image
            model_input = torch.cat([noisy_next_img, current_img_batch], dim=1) # [B, 6, H, W]

            # Forward pass with optional mixed precision
            if USE_AMP and scaler is not None:
                with torch.amp.autocast('cuda'):
                    predicted_noise_epsilon = model(model_input, t)
                    loss = criterion(predicted_noise_epsilon, added_noise_epsilon)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                predicted_noise_epsilon = model(model_input, t)
                loss = criterion(predicted_noise_epsilon, added_noise_epsilon)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
            # Memory cleanup every 10 batches
            if batch_idx % 10 == 0:
                gc.collect()
                if DEVICE.type == 'mps':
                    torch.mps.empty_cache()

            # TensorBoard logging
            if global_step % LOG_FREQ == 0:
                # Log scalar metrics
                tb_writer.add_scalar('Loss/Batch', loss.item(), global_step)
                tb_writer.add_scalar('Loss/Running_Average', total_loss/num_batches, global_step)
                tb_writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                
                # Log gradient norms
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                tb_writer.add_scalar('Gradients/Total_Norm', total_norm, global_step)
                
                # Log sample images periodically
                if LOG_IMAGES and global_step % (LOG_FREQ * 10) == 0:
                    with torch.no_grad():
                        log_images_to_tensorboard(
                            tb_writer, 
                            current_img_batch, 
                            next_img_batch, 
                            noisy_next_img,
                            predicted_noise_epsilon, 
                            added_noise_epsilon, 
                            global_step
                        )

            global_step += 1

            # Update batch progress bar with current loss
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss/num_batches:.6f}'
            })

        # Calculate epoch statistics
        avg_epoch_loss = total_loss / num_batches
        loss_history.append(avg_epoch_loss)
        
        # Log epoch metrics to TensorBoard
        tb_writer.add_scalar('Loss/Epoch', avg_epoch_loss, epoch)
        
        # Check if this is the best model
        is_best = SAVE_BEST and avg_epoch_loss < best_loss
        if is_best:
            best_loss = avg_epoch_loss
            tb_writer.add_scalar('Loss/Best', best_loss, epoch)

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Avg Loss': f'{avg_epoch_loss:.6f}',
            'Best Loss': f'{best_loss:.6f}',
            'Best': '✓' if is_best else ''
        })

        # Print epoch summary with elapsed time
        elapsed_time = time.time() - training_start_time
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        status = "*** NEW BEST ***" if is_best else ""
        tqdm.write(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_epoch_loss:.6f} - Elapsed: {elapsed_str} {status}")

        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_FREQ == 0 or (epoch + 1) == EPOCHS or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_epoch_loss,
                loss_history=loss_history,
                is_best=is_best
            )

    # Close progress bars and TensorBoard writer
    epoch_pbar.close()
    tb_writer.close()
    
    # Final training summary
    total_training_time = time.time() - training_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_training_time)))
    
    print("Training complete.")
    print(f"Total training time: {total_time_str}")
    print(f"Best loss achieved: {best_loss:.6f}")
    print(f"All checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"TensorBoard logs saved to: {TENSORBOARD_LOG_DIR}")
    print(f"View logs with: tensorboard --logdir={TENSORBOARD_LOG_DIR}")
