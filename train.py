#!/usr/bin/env python3
"""
Training script for MB-sMIL model with trainable UNIv2.

- Two-step training: MIL predictor + UNI fine-tuning
- Attention-based patch selection for UNI fine-tuning
- Memory optimizations

Usage:
    # Single GPU
    torchrun --nproc_per_node=1 train.py

    # Multi-GPU (e.g., 4 GPUs)
    torchrun --nproc_per_node=4 train.py
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import f1_score
from tqdm import tqdm
from huggingface_hub import login

from models.uni_model import load_uni_model
from models.mil_model import create_mil_model
from data.dataset import PatchDataset, create_data_splits
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    save_checkpoint,
)
from utils.metrics import evaluate, print_metrics


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MIL-PSC model with trainable UNI"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    return parser.parse_args()


def finetune_uni_with_attention(
    attention_weights,
    labels,
    slide_id,
    slide_dir,
    model_uni,
    mil_predictor,
    optimizer_uni,
    criterion_uni,
    config,
):
    """
    Fine-tune UNI model using k most attended patches.

    This function:
    1. Selects top-k patches based on attention weights
    2. Creates a dataset with these patches (with augmentation)
    3. Fine-tunes the UNI model on these patches

    Args:
        attention_weights: Attention weights from MIL model [N, num_classes]
        labels: Ground truth labels
        slide_id: Slide identifier
        slide_dir: Directory containing patches
        model_uni: UNI model
        mil_predictor: MB-sMIL predictor model
        optimizer_uni: Optimizer for UNI
        criterion_uni: Loss function for UNI
        config: Configuration dictionary
    """
    rank = get_rank()
    world_size = get_world_size()
    batch_size = config["training"]["batch_size"]
    k_samples = config["training"].get("finetune_top_k", 20)

    # Select top-k patches based on maximum attention across classes
    with torch.no_grad():
        attention_scores = torch.max(attention_weights, dim=-1).values
        top_k_indices = torch.topk(
            attention_scores, min(k_samples, len(attention_scores))
        ).indices
        top_k_indices = torch.clamp(top_k_indices, max=len(attention_scores) - 5)

    # Create fine-tuning dataset with selected patches
    dataset = PatchDataset(
        slide_dir,
        slide_id,
        config,
        selected_indices=top_k_indices,
        is_selected=True,
        augment=True,
    )

    patch_sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    pin_memory = config["training"].get("pin_memory", True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=patch_sampler,
        num_workers=config["training"]["num_workers"],
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Fine-tune UNI model
    model_uni.train()
    mil_predictor.eval()

    # Extract features with gradients
    all_features = []
    for batch_indices, patches in loader:
        batch_features = model_uni(patches.to(rank))
        all_features.append(batch_features.cpu())
        del patches, batch_features

    all_features = torch.cat(all_features, dim=0).to(rank)

    # Gather features from all GPUs
    gathered_features = [
        torch.zeros_like(all_features).to(rank) for _ in range(world_size)
    ]
    dist.all_gather(gathered_features, all_features)
    all_features = torch.cat(gathered_features, dim=0).to(rank)
    del gathered_features

    # Forward through MIL predictor and compute loss
    logits, _ = mil_predictor(all_features)
    loss = criterion_uni(logits, labels.argmax(dim=1))

    # Backward pass for UNI
    optimizer_uni.zero_grad()
    loss.backward()
    optimizer_uni.step()

    # Set models back to appropriate modes
    model_uni.eval()
    mil_predictor.train()

    # Memory cleanup
    torch.cuda.empty_cache()


def train_epoch(
    slide_dir,
    train_labels_dict,
    model_uni,
    mil_predictor,
    optimizer,
    optimizer_uni,
    criterion,
    criterion_uni,
    config,
    epoch,
):
    """
    Train for one epoch with two-step training:
    1. Train MIL predictor with frozen UNI
    2. Fine-tune UNI with top-k attended patches

    Args:
        slide_dir: Directory containing slide patches
        train_labels_dict: Dictionary mapping slide IDs to labels
        model_uni: UNI foundation model
        mil_predictor: MIL predictor model
        optimizer: Optimizer for MIL predictor
        optimizer_uni: Optimizer for UNI
        criterion: Loss function for MIL predictor
        criterion_uni: Loss function for UNI
        config: Configuration dictionary
        epoch: Current epoch number

    Returns:
        avg_loss: Average training loss
        f1: Macro F1 score
    """
    rank = get_rank()
    world_size = get_world_size()
    batch_size = config["training"]["batch_size"]

    model_uni.eval()  # Start with UNI in eval mode
    mil_predictor.train()

    running_loss = 0
    all_preds = []
    all_labels = []

    if is_main_process():
        pbar = tqdm(train_labels_dict.items())
        pbar.set_description(f"Epoch {epoch + 1}/{config['training']['epochs']}")
    else:
        pbar = train_labels_dict.items()

    for i, (slide_id, labels) in enumerate(pbar, 1):
        labels = labels.unsqueeze(0).to(rank)

        # STEP 1: Train MIL predictor with frozen UNI
        full_dataset = PatchDataset(
            slide_dir, slide_id, config, augment=True, is_shuffle=False
        )
        patch_sampler = DistributedSampler(
            full_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        pin_memory = config["training"].get("pin_memory", True)
        loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=patch_sampler,
            num_workers=config["training"]["num_workers"],
            pin_memory=pin_memory,
            drop_last=False,
        )

        # Extract features (no gradients needed for UNI in this step)
        all_features = []
        with torch.no_grad():
            for batch_indices, patches in loader:
                batch_features = model_uni(patches.to(rank))
                all_features.append(batch_features.cpu())
                del patches, batch_features

        all_features = torch.cat(all_features, dim=0).to(rank)

        # Gather features from all GPUs
        gathered_features = [
            torch.zeros_like(all_features).to(rank) for _ in range(world_size)
        ]
        dist.all_gather(gathered_features, all_features)
        all_features = torch.cat(gathered_features, dim=0).to(rank)
        del gathered_features

        # Forward pass through MIL model
        logits, attention_weights = mil_predictor(all_features)
        loss = criterion(logits, labels.argmax(dim=1))

        # Backward pass for MIL predictor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dist.barrier()

        # Update metrics
        running_loss = (running_loss * (i - 1) + loss.item()) / i
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels.argmax(dim=1).cpu())
        epoch_f1 = f1_score(
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(), average="macro"
        )

        if is_main_process():
            pbar.set_postfix({"loss": f"{running_loss:.4f}", "f1": f"{epoch_f1:.4f}"})

        # STEP 2: Fine-tune UNI with attention-selected patches
        finetune_uni_with_attention(
            attention_weights.detach(),
            labels,
            slide_id,
            slide_dir,
            model_uni,
            mil_predictor,
            optimizer_uni,
            criterion_uni,
            config,
        )

        dist.barrier()

        # Ensure correct model states
        model_uni.eval()
        mil_predictor.train()

    if is_main_process():
        pbar.close()

    return running_loss, epoch_f1


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup distributed training
    setup_distributed(config)
    rank = get_rank()

    # Login to Hugging Face
    if is_main_process():
        login(config["huggingface"]["token"])
        print("===================================")
        print(" " * 15 + "Training MB-sMIL with trainable UNIv2)")
        print("===================================")

    # Create output directories
    if is_main_process():
        os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
        os.makedirs(config["output"]["log_dir"], exist_ok=True)

    # Load data splits
    slide_dir = config["data"]["slide_dir"]
    if is_main_process():
        print("\nCreating data splits...")
    train_labels_dict, val_labels_dict, test_labels_dict = create_data_splits(
        slide_dir, config
    )

    # Load models
    if is_main_process():
        print("\nLoading models...")
    model_uni = load_uni_model(config).to(rank)
    mil_predictor = create_mil_model(config).to(rank)

    # Wrap models with DDP
    model_uni = DDP(model_uni, device_ids=[rank], find_unused_parameters=False)
    mil_predictor = DDP(mil_predictor, device_ids=[rank], find_unused_parameters=False)

    # Setup optimizers and losses
    optimizer_mil = torch.optim.Adam(
        mil_predictor.parameters(), lr=config["training"]["learning_rate"]
    )
    optimizer_uni = torch.optim.Adam(
        model_uni.parameters(), lr=config["training"]["learning_rate"]
    )

    class_weights = torch.tensor(config["training"]["class_weights"]).cuda(rank)
    criterion_mil = nn.CrossEntropyLoss(weight=class_weights).to(rank)
    criterion_uni = nn.CrossEntropyLoss(weight=class_weights).to(rank)

    if is_main_process():
        print(f"\nTraining configuration:")
        print(f"   Epochs: {config['training']['epochs']}")
        print(f"   Learning rate: {config['training']['learning_rate']}")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   Class weights: {config['training']['class_weights']}")
        print(f"   Fine-tune top-k: {config['training'].get('finetune_top_k', 20)}")
        print(
            f"   Max patches/slide: {config['data'].get('max_patches_per_slide', 'unlimited')}"
        )

    # Training loop
    best_f1 = 0
    epochs = config["training"]["epochs"]

    if is_main_process():
        print("\n===================================")
        print(" " * 25 + "Starting Training")
        print("===================================")

    for epoch in range(epochs):
        if is_main_process():
            print("\n===================================")
            print(f"Epoch {epoch + 1}/{epochs}")
            print("===================================")

        # Train
        train_loss, train_f1 = train_epoch(
            slide_dir,
            train_labels_dict,
            model_uni,
            mil_predictor,
            optimizer_mil,
            optimizer_uni,
            criterion_mil,
            criterion_uni,
            config,
            epoch,
        )

        if is_main_process():
            print(f"\nTrain - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")

        # Validate
        val_loss, val_f1, cm, val_auc = evaluate(
            slide_dir, val_labels_dict, model_uni, mil_predictor, criterion_mil, config
        )

        if is_main_process():
            print(
                f"Validation - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}"
            )
            print_metrics(cm, auc=val_auc, prefix="Validation")

            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                print(f"\nBest model updated! F1: {best_f1:.4f}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "uni_model_state_dict": model_uni.state_dict(),
                        "mil_predictor_state_dict": mil_predictor.state_dict(),
                        "optimizer_mil_state_dict": optimizer_mil.state_dict(),
                        "optimizer_uni_state_dict": optimizer_uni.state_dict(),
                        "best_f1": best_f1,
                        "config": config,
                    },
                    filename=os.path.join(
                        config["output"]["checkpoint_dir"], "checkpoint.pth"
                    ),
                )

            # Save last model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "uni_model_state_dict": model_uni.state_dict(),
                    "mil_predictor_state_dict": mil_predictor.state_dict(),
                    "optimizer_mil_state_dict": optimizer_mil.state_dict(),
                    "optimizer_uni_state_dict": optimizer_uni.state_dict(),
                    "val_f1": val_f1,
                    "config": config,
                },
                filename=os.path.join(
                    config["output"]["checkpoint_dir"], "last_checkpoint.pth"
                ),
            )

    dist.barrier()
    torch.cuda.empty_cache()

    # Test with last model
    if is_main_process():
        print("\n===================================")
        print("Testing Last Model")
        print("===================================")

    test_loss, test_f1, cm, test_auc = evaluate(
        slide_dir, test_labels_dict, model_uni, mil_predictor, criterion_mil, config
    )

    if is_main_process():
        print(
            f"\nTest (Last) - Loss: {test_loss:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}"
        )
        print_metrics(cm, auc=test_auc, prefix="Test")

    # Test with best model
    if is_main_process():
        print("\n===================================")
        print("Testing Best Model")
        print("===================================")

    checkpoint = torch.load(
        os.path.join(config["output"]["checkpoint_dir"], "checkpoint.pth"),
        map_location=f"cuda:{rank}",
    )
    model_uni.load_state_dict(checkpoint["uni_model_state_dict"])
    mil_predictor.load_state_dict(checkpoint["mil_predictor_state_dict"])

    test_loss, test_f1, cm, test_auc = evaluate(
        slide_dir, test_labels_dict, model_uni, mil_predictor, criterion_mil, config
    )

    if is_main_process():
        print(
            f"\nTest (Best) - Loss: {test_loss:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}"
        )
        print_metrics(cm, auc=test_auc, prefix="Test")

        print("\n===================================")
        print(" " * 22 + "Training Completed!")
        print("===================================")
        print(f"\nBest validation F1: {best_f1:.4f}")
        print(f"Checkpoints saved in: {config['output']['checkpoint_dir']}")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
