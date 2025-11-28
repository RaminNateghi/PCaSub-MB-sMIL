import torch
import torch.distributed as dist
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.dataset import PatchDataset
from utils.distributed import get_rank, get_world_size, is_main_process


def evaluate(slide_dir, labels_dict, model_uni, mil_predictor, criterion, config):
    """
    Evaluate model on a dataset.

    This function processes each slide by:
    1. Loading all patches across distributed GPUs
    2. Extracting features with UNI model
    3. Gathering features from all GPUs
    4. Making slide-level prediction with MIL model

    Args:
        slide_dir: Directory containing slide patches
        labels_dict: Dictionary mapping slide IDs to labels
        model_uni: UNIv2 foundation model
        mil_predictor:  Multi-branch Multiple Instance Learning model with self-attention mechanism (MB-sMIL)
        criterion: Loss function
        config: Configuration dictionary

    Returns:
        avg_loss: Average loss over dataset
        f1: Macro F1 score
        cm: Confusion matrix
        auc: ROC AUC score
    """
    rank = get_rank()
    world_size = get_world_size()
    batch_size = config["training"]["batch_size"]
    num_classes = config["data"]["num_classes"]

    model_uni.eval()
    mil_predictor.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_preds_probs = []  # For AUC calculation
    all_labels_onehot = []  # For AUC calculation

    if is_main_process():
        pbar = tqdm(labels_dict.items())
        pbar.set_description("Evaluating")
    else:
        pbar = labels_dict.items()

    with torch.no_grad():
        for i, (slide_id, labels) in enumerate(pbar, 1):
            labels = labels.unsqueeze(0).to(rank)

            # Create dataset for this slide
            full_dataset = PatchDataset(
                slide_dir, slide_id, config, augment=False, is_shuffle=False
            )
            patch_sampler = DistributedSampler(
                full_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )

            # Use pin_memory setting from config
            pin_memory = config["training"].get("pin_memory", True)

            loader = DataLoader(
                full_dataset,
                batch_size=batch_size,
                sampler=patch_sampler,
                num_workers=config["training"]["num_workers"],
                pin_memory=pin_memory,
                drop_last=False,
            )

            # Extract features from patches
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

            # Concatenate features from all GPUs
            all_features = torch.cat(gathered_features, dim=0).to(rank)
            del gathered_features

            # Make slide-level prediction
            logits, attention_weights = mil_predictor(all_features)
            loss = criterion(logits, labels.argmax(dim=1))

            # Update running metrics
            running_loss = (running_loss * (i - 1) + loss.item()) / i

            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels.argmax(dim=1).cpu())

            # Store probabilities and one-hot labels for AUC
            all_preds_probs.append(torch.softmax(logits, dim=1).cpu())
            all_labels_onehot.append(labels.cpu())

            # Calculate F1 score (use micro for consistency with your code)
            epoch_f1 = f1_score(
                torch.cat(all_labels).numpy(),
                torch.cat(all_preds).numpy(),
                average="micro",
            )

            if is_main_process():
                pbar.set_postfix(
                    {"loss": f"{running_loss:.4f}", "f1": f"{epoch_f1:.4f}"}
                )

    if is_main_process():
        pbar.close()

    # Compute confusion matrix
    cm = confusion_matrix(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy())

    # Compute AUC score
    try:
        auc = roc_auc_score(
            torch.cat(all_labels_onehot, 0).numpy(),
            torch.cat(all_preds_probs, 0).numpy(),
            multi_class="ovo",
            average="micro",
        )
    except:
        auc = 0.0  # If AUC cannot be computed (e.g., missing classes)

    return running_loss, epoch_f1, cm, auc


def print_metrics(cm, auc=None, prefix=""):
    """
    Print confusion matrix and derived metrics.

    Args:
        cm: Confusion matrix (numpy array)
        auc: ROC AUC score (optional)
        prefix: Prefix for print statements (e.g., "Train", "Val", "Test")
    """
    if is_main_process():
        print(f"\n{prefix} Confusion Matrix:")
        print(cm)

        # Calculate accuracy, precision, recall from confusion matrix
        y_true = []
        y_pred = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                y_true.extend([i] * cm[i][j])
                y_pred.extend([j] * cm[i][j])

        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

            print(f"{prefix} Accuracy: {acc:.4f}")
            print(f"{prefix} Precision (macro): {precision:.4f}")
            print(f"{prefix} Recall (macro): {recall:.4f}")
            print(f"{prefix} F1 (macro): {f1:.4f}")

            if auc is not None:
                print(f"{prefix} AUC (micro): {auc:.4f}")
