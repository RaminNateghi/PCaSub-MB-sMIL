#!/usr/bin/env python3
"""
Testing script for MB-sMIL model with trainable UNIv2.

Usage:
    # Test on test set with best checkpoint
    torchrun --nproc_per_node=4 test.py --checkpoint ./checkpoints/checkpoint.pth

    # Test on validation set
    torchrun --nproc_per_node=4 test.py --checkpoint ./checkpoints/checkpoint.pth --split val

    # Test on training set
    torchrun --nproc_per_node=1 test.py --checkpoint ./checkpoints/checkpoint.pth --split train
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from huggingface_hub import login

from models.uni_model import load_uni_model
from models.mil_model import create_mil_model
from data.dataset import create_data_splits
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    get_rank,
    is_main_process,
    load_checkpoint,
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
        description="Test MIL-PSC model with trainable UNI"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/checkpoint.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on",
    )
    return parser.parse_args()


def main():
    """Main testing function."""
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
        print(" " * 17 + "Testing MB-sMIL (Trainable UNIv2)")
        print("===================================")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Split: {args.split}")
        print("===================================")

    # Load data splits
    slide_dir = config["data"]["slide_dir"]
    if is_main_process():
        print("\nCreating data splits...")
    train_labels_dict, val_labels_dict, test_labels_dict = create_data_splits(
        slide_dir, config
    )

    # Select the appropriate split
    if args.split == "train":
        labels_dict = train_labels_dict
    elif args.split == "val":
        labels_dict = val_labels_dict
    else:
        labels_dict = test_labels_dict

    if is_main_process():
        print(f"\nEvaluating on {len(labels_dict)} slides from {args.split} set")

    # Load models
    if is_main_process():
        print("\nLoading models...")
    model_uni = load_uni_model(config).to(rank)
    mil_predictor = create_mil_model(config).to(rank)

    # Wrap models with DDP
    model_uni = DDP(model_uni, device_ids=[rank], find_unused_parameters=False)
    mil_predictor = DDP(mil_predictor, device_ids=[rank], find_unused_parameters=False)

    # Load checkpoint
    if is_main_process():
        print(f"\nLoading checkpoint from {args.checkpoint}...")

    checkpoint = load_checkpoint(args.checkpoint, device=f"cuda:{rank}")

    # Load both UNI and MIL predictor weights
    model_uni.load_state_dict(checkpoint["uni_model_state_dict"])
    mil_predictor.load_state_dict(checkpoint["mil_predictor_state_dict"])

    if is_main_process():
        epoch = checkpoint.get("epoch", "N/A")
        best_f1 = checkpoint.get("best_f1", checkpoint.get("val_f1", "N/A"))
        print(f"   Checkpoint epoch: {epoch}")
        if isinstance(best_f1, float):
            print(f"   Best F1 (from training): {best_f1:.4f}")

    # Setup loss
    class_weights = torch.tensor(config["training"]["class_weights"]).cuda(rank)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(rank)

    # Evaluate
    if is_main_process():
        print(f"\nEvaluating on {args.split} set...")
        print("===================================")

    test_loss, test_f1, cm, test_auc = evaluate(
        slide_dir, labels_dict, model_uni, mil_predictor, criterion, config
    )

    if is_main_process():
        print("\n===================================")
        print(" " * 28 + "RESULTS")
        print("===================================")
        print(f"\n{args.split.upper()} SET PERFORMANCE:")
        print(f"   Loss: {test_loss:.4f}")
        print(f"   F1 Score (micro): {test_f1:.4f}")
        print(f"   AUC Score (micro): {test_auc:.4f}")
        print_metrics(cm, auc=test_auc, prefix=args.split.upper())
        print("\n===================================")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
