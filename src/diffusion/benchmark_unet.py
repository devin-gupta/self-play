import argparse
import time
import torch
import os
from pathlib import Path

# Assuming this script is in src/diffusion/, and model.py is in the same directory
# and dataloader.py is in src/data_utils/
from .model import ConditionalUNet
from ..data_utils.dataloader import KitchenPairDataset
from torch.utils.data import DataLoader

def get_model_size_info(model: torch.nn.Module) -> tuple[float, int]:
    """Calculates model size in MB and number of parameters.

    Args:
        model: The PyTorch model.

    Returns:
        A tuple containing:
            - size_all_mb (float): Total model size in megabytes.
            - num_params (int): Total number of parameters.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Count only trainable parameters
    return size_all_mb, num_params

def main(args: argparse.Namespace) -> None:
    """Main function to run the benchmark."""
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Initialize Model
    print("Initializing model...")
    # Ensure channel_mults and attention_resolutions are correctly parsed
    channel_mults_tuple = tuple(map(int, args.model_channel_mults.split(',')))
    attention_resolutions_tuple = tuple(map(int, args.model_attention_resolutions.split(',')))

    model = ConditionalUNet(
        in_img_channels=6,  # 3 for noisy_next_img (proxy: future_frame), 3 for current_img (current_frame)
        out_img_channels=3, # For predicted noise
        base_channels=args.model_base_channels,
        channel_mults=channel_mults_tuple,
        num_res_blocks_per_level=args.model_num_res_blocks,
        attention_resolutions=attention_resolutions_tuple,
        dropout_rate=args.model_dropout_rate, # Added dropout_rate
        time_emb_dim=args.model_time_emb_dim,
        time_emb_mlp_dim=args.model_time_emb_mlp_dim,
        initial_img_resolution=args.target_size_dataloader # Critical: must match data
    ).to(device)
    model.eval()

    model_size_mb, num_params = get_model_size_info(model)
    print(f"Model initialized: {num_params:,} trainable parameters, Size: {model_size_mb:.2f} MB")

    if args.model_checkpoint_path:
        checkpoint_path = Path(args.model_checkpoint_path)
        if checkpoint_path.exists():
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print(f"Loaded model weights from {args.model_checkpoint_path}")
            except Exception as e:
                print(f"Error loading checkpoint {args.model_checkpoint_path}: {e}. Using random weights.")
        else:
            print(f"Warning: Checkpoint path {args.model_checkpoint_path} not found. Using random weights.")

    # 2. Initialize DataLoader
    print(f"Loading dataset with k={args.k_dataloader}, target_size={args.target_size_dataloader}x{args.target_size_dataloader} from {args.data_dir_dataloader}")
    try:
        dataset = KitchenPairDataset(
            k_step_future=args.k_dataloader,
            data_dir=args.data_dir_dataloader,
            target_size=args.target_size_dataloader,
            force_rebuild=args.force_rebuild_dataloader 
        )
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        print("Please ensure the dataset is pre-generated using src/data_utils/dataloader.py")
        print(f"Example: python src/data_utils/dataloader.py --k {args.k_dataloader} --target_size {args.target_size_dataloader} --data_dir {args.data_dir_dataloader}")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Please generate data first using src/data_utils/dataloader.py.")
        return
    
    if args.num_batches > (len(dataset) // args.batch_size) and (len(dataset) % args.batch_size !=0) :
         print(f"Warning: num_batches ({args.num_batches}) is greater than the number of available batches in the dataset ({len(dataset)//args.batch_size}).")
         print(f"Will run for a maximum of {len(dataset)//args.batch_size} batches.")


    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle for benchmarking
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        drop_last=True # Drop last incomplete batch for consistent batch sizes in benchmark
    )
    
    if len(dataloader) == 0:
        print(f"Dataloader is empty. This might be due to batch_size ({args.batch_size}) being larger than dataset size ({len(dataset)}).")
        return


    # 3. Benchmarking loop
    total_time_seconds = 0.0
    total_items_processed = 0

    # Warm-up runs
    if args.num_warmup_batches > 0:
        print(f"Running {args.num_warmup_batches} warm-up batches...")
        with torch.no_grad():
            warmup_batches_done = 0
            for batch_idx, batch_data in enumerate(dataloader):
                if warmup_batches_done >= args.num_warmup_batches:
                    break
                current_frames, future_frames, _, episode_current_steps, _ = batch_data
                
                x_concat = torch.cat((future_frames, current_frames), dim=1).to(device)
                # Model expects time as [B,1] or [B], SinusoidalPositionalEncoding handles [B] and unsqueezes.
                time_input = episode_current_steps.float().to(device)


                _ = model(x_concat, time_input)
                warmup_batches_done +=1
            
            if 'cuda' in args.device:
                torch.cuda.synchronize()
        print(f"Warm-up complete ({warmup_batches_done} batches run).")

    print(f"Running benchmark for up to {args.num_batches} batches...")
    actual_batches_run = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if actual_batches_run >= args.num_batches:
                break
            
            current_frames, future_frames, _, episode_current_steps, _ = batch_data
            
            x_concat = torch.cat((future_frames, current_frames), dim=1).to(device)
            time_input = episode_current_steps.float().to(device) # [B]

            start_event, end_event = None, None
            if 'cuda' in args.device:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                cpu_start_time = time.perf_counter()

            _ = model(x_concat, time_input)

            batch_time_seconds = 0
            if 'cuda' in args.device:
                end_event.record()
                torch.cuda.synchronize() # Wait for the events to complete
                batch_time_seconds = start_event.elapsed_time(end_event) / 1000.0  # elapsed_time returns ms
            else:
                cpu_end_time = time.perf_counter()
                batch_time_seconds = cpu_end_time - cpu_start_time
            
            total_time_seconds += batch_time_seconds
            total_items_processed += x_concat.size(0)
            actual_batches_run += 1
            
            if (batch_idx + 1) % args.print_freq == 0:
                 print(f"Batch {actual_batches_run}/{args.num_batches}: Time {batch_time_seconds:.4f}s")
    
    if actual_batches_run == 0:
        print("No batches were run for benchmarking. Check dataset, num_batches, or batch_size arguments.")
        return

    avg_time_per_batch = total_time_seconds / actual_batches_run
    images_per_second = total_items_processed / total_time_seconds if total_time_seconds > 0 else 0

    print("\n--- FP32 Model Benchmark Results ---")
    print(f"Device: {args.device}")
    print(f"Model Trainable Parameters: {num_params:,}")
    print(f"Model Size (parameters + buffers): {model_size_mb:.2f} MB")
    print(f"Input Image Resolution: {args.target_size_dataloader}x{args.target_size_dataloader}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Batches Tested: {actual_batches_run}")
    print(f"Average Inference Time per Batch: {avg_time_per_batch:.4f} seconds")
    print(f"Throughput: {images_per_second:.2f} images/second")
    print("------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ConditionalUNet FP32 Performance")
    
    # --- Dataset Arguments ---
    parser.add_argument("--k_dataloader", type=int, default=10, help="k_step_future for KitchenPairDataset.")
    parser.add_argument("--target_size_dataloader", type=int, default=256, help="Target image size for KitchenPairDataset (e.g., 256 for 256x256). This also sets the model's initial_img_resolution.")
    parser.add_argument("--data_dir_dataloader", type=str, default="data/kitchen_pairs", help="Data directory for KitchenPairDataset.")
    parser.add_argument("--force_rebuild_dataloader", action="store_true", help="Force rebuild the dataset for the dataloader.")
    
    # --- Model Structure Arguments ---
    # These must match the architecture of the model defined in model.py or loaded from a checkpoint.
    # Defaults are based on src/diffusion/model.py ConditionalUNet defaults or sensible values for 256px.
    parser.add_argument("--model_base_channels", type=int, default=64, help="Base channels for the U-Net.")
    parser.add_argument("--model_channel_mults", type=str, default="1,2,3,4", help="Channel multipliers for U-Net levels (comma-separated string, e.g., '1,2,4,8').")
    parser.add_argument("--model_num_res_blocks", type=int, default=2, help="Number of residual blocks per U-Net level.")
    parser.add_argument("--model_attention_resolutions", type=str, default="32,16", help="Resolutions at which to apply attention (comma-separated string, e.g., '32,16' for a 256px input -> 256,128,64,32,16). Model default is (120,60) for 480px input.")
    parser.add_argument("--model_dropout_rate", type=float, default=0.1, help="Dropout rate for ResidualBlocks.")
    parser.add_argument("--model_time_emb_dim", type=int, default=256, help="Dimension of sinusoidal time embedding.")
    parser.add_argument("--model_time_emb_mlp_dim", type=int, default=1024, help="Dimension of MLP for time embedding.")
    parser.add_argument("--model_checkpoint_path", type=str, default=None, help="Optional path to model checkpoint (.pth or .pt) for loading pre-trained weights.")

    # --- Benchmark Execution Arguments ---
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for benchmarking.")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of batches to run for benchmark. If more than available in dataset, runs on all available.")
    parser.add_argument("--num_warmup_batches", type=int, default=10, help="Number of warm-up batches before starting timing.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", help="Device to use: 'cpu', 'cuda', or 'mps'.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers. Set to 0 for simplicity/debugging, >0 for potential speedup if I/O is a bottleneck.")
    parser.add_argument("--print_freq", type=int, default=10, help="Frequency of printing batch processing time (e.g., every 10 batches).")

    parsed_args = parser.parse_args()
    
    # Validate attention resolutions based on target_size and channel_mults
    try:
        parsed_attention_resolutions = tuple(map(int, parsed_args.model_attention_resolutions.split(',')))
        current_res = parsed_args.target_size_dataloader
        possible_resolutions = {current_res}
        for _ in range(len(parsed_args.model_channel_mults.split(',')) -1): # num downsamples
            current_res //= 2
            possible_resolutions.add(current_res)
        
        for attn_res in parsed_attention_resolutions:
            if attn_res not in possible_resolutions:
                print(f"Warning: Attention resolution {attn_res} is not a valid downsampled resolution for initial size {parsed_args.target_size_dataloader}.")
                print(f"Possible resolutions are: {sorted(list(possible_resolutions))}")
    except ValueError:
        print("Error: --model_attention_resolutions must be a comma-separated list of integers.")
        exit(1)

    main(parsed_args) 