"""
utils.py
"""
import torch


def get_linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    Returns a linear beta schedule from beta_start to beta_end.
    """

    return torch.linspace(beta_start, beta_end, timesteps)


def add_noise_to_images(original_images, t, device, precomputed=None):
    """Adds noise to images x_0 to get x_t."""
    # original_images: [B, C, H, W]
    # t: [B] tensor of timesteps
    batch_size = original_images.shape[0]
    noise = torch.randn_like(original_images, device=device) # epsilon

    # Gather coefficients for each t in the batch
    sqrt_alphas_cumprod_t = precomputed["sqrt_alphas_cumprod"][t].to(device).view(batch_size, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = precomputed["sqrt_one_minus_alphas_cumprod"][t].to(device).view(batch_size, 1, 1, 1)

    noisy_images = sqrt_alphas_cumprod_t * original_images + \
                   sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_images, noise # Return noisy image and the noise that was added (target)