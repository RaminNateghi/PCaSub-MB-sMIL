import os
import torch
import torch.distributed as dist


def setup_distributed(config):
    """
    Initialize distributed training environment.

    Sets up the distributed process group and configures environment variables
    for NCCL backend.

    Args:
        config: Configuration dictionary
    """
    # Set environment variables from config
    dist_config = config["distributed"]
    os.environ["NCCL_DEBUG"] = dist_config["nccl_debug"]
    os.environ["NCCL_P2P_DISABLE"] = dist_config["nccl_p2p_disable"]

    # Initialize process group
    dist.init_process_group(backend=dist_config["backend"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed():
    """Clean up distributed training environment."""
    dist.destroy_process_group()


def get_rank():
    """
    Get the rank of the current process.

    Returns:
        int: Rank of current process (0 for single GPU)
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size():
    """
    Get the total number of processes.

    Returns:
        int: Number of GPUs/processes
    """
    return torch.cuda.device_count()


def is_main_process():
    """
    Check if this is the main process (rank 0).

    Returns:
        bool: True if this is rank 0
    """
    return get_rank() == 0


def save_checkpoint(state, filename):
    """
    Save checkpoint (only on main process).

    Args:
        state: Dictionary containing model state and other info
        filename: Path to save checkpoint
    """
    if is_main_process():
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename, device):
    """
    Load checkpoint.

    Args:
        filename: Path to checkpoint file
        device: Device to load checkpoint to (e.g., 'cuda:0')

    Returns:
        dict: Checkpoint dictionary
    """
    checkpoint = torch.load(filename, map_location=device)
    print(f"Checkpoint loaded from {filename}")
    return checkpoint
