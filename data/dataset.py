import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit


class PatchDataset(Dataset):
    """
    Dataset for loading image patches with optional augmentation.

    This dataset loads individual patches from whole slide images and applies
    preprocessing and augmentation as specified.
    """

    def __init__(
        self,
        slide_dir,
        slide_id,
        config,
        selected_indices=None,
        is_selected=False,
        return_paths=False,
        augment=False,
        is_shuffle=False,
    ):
        """
        Args:
            slide_dir: Directory containing patch files
            slide_id: ID of the slide to load patches from
            config: Configuration dictionary
            selected_indices: Optional list of indices to select specific patches
            is_selected: Whether to use selected_indices
            return_paths: Whether to return patch paths (currently unused)
            augment: Whether to apply data augmentation
            is_shuffle: Whether to shuffle the patch list
        """
        self.slide_dir = slide_dir
        self.slide_id = slide_id
        self.is_selected = is_selected
        self.return_paths = return_paths
        self.augment = augment
        self.config = config

        # Get all patch files for this slide
        self.patch_files = [f for f in os.listdir(slide_dir) if f.startswith(slide_id)]

        # Random sampling if max_patches_per_slide is set and not using selected_indices
        if not is_selected and "max_patches_per_slide" in config["data"]:
            if config["data"]["max_patches_per_slide"] is not None:
                max_patches = config["data"]["max_patches_per_slide"]
                if max_patches is not None and len(self.patch_files) > max_patches:
                    random_selection = random.sample(
                        range(len(self.patch_files)), max_patches
                    )
                    self.patch_files = [self.patch_files[i] for i in random_selection]

        if is_selected and selected_indices is not None:
            self.patch_files = [self.patch_files[i] for i in selected_indices]

        if is_shuffle:
            random.shuffle(self.patch_files)

        # Get normalization constants from config
        norm_mean = config["normalization"]["mean"]
        norm_std = config["normalization"]["std"]
        img_size = config["model"]["uni"]["img_size"]

        # Augmentation pipeline
        if augment:
            aug_config = config["training"]["augmentation"]
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=aug_config["rotation_degrees"]),
                    transforms.ColorJitter(
                        brightness=aug_config["color_jitter"]["brightness"],
                        contrast=aug_config["color_jitter"]["contrast"],
                        saturation=aug_config["color_jitter"]["saturation"],
                        hue=aug_config["color_jitter"]["hue"],
                    ),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=tuple(aug_config["affine"]["translate"]),
                        scale=tuple(aug_config["affine"]["scale"]),
                        shear=aug_config["affine"]["shear"],
                    ),
                    transforms.RandomPerspective(
                        distortion_scale=aug_config["perspective"]["distortion_scale"],
                        p=aug_config["perspective"]["p"],
                    ),
                    transforms.GaussianBlur(
                        kernel_size=aug_config["gaussian_blur"]["kernel_size"],
                        sigma=tuple(aug_config["gaussian_blur"]["sigma"]),
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )
        else:
            # Basic preprocessing pipeline (no augmentation)
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        """
        Get a patch by index.

        Returns:
            idx: Index of the patch
            patch: Preprocessed patch tensor
        """
        patch_file = self.patch_files[idx]
        patch_path = os.path.join(self.slide_dir, patch_file)
        patch = self.preprocess(Image.open(patch_path))
        return idx, patch


def create_data_splits(slide_dir, config):
    """
    Create train/val/test splits from slide directory.

    Splits are performed at the patient level to prevent data leakage.
    The filename format expected is: {patient_id}_{slide_info}-{patch_info}_{class}.png

    Args:
        slide_dir: Directory containing slide patches
        config: Configuration dictionary

    Returns:
        train_labels_dict: Dictionary mapping slide IDs to labels for training
        val_labels_dict: Dictionary mapping slide IDs to labels for validation
        test_labels_dict: Dictionary mapping slide IDs to labels for testing
    """
    num_classes = config["data"]["num_classes"]

    # Create labels dictionary from filenames
    # Filename format: {patient_id}_{slide_info}-{patch_info}_{class}.png
    labels_dict = {
        si.split("-")[0]: torch.nn.functional.one_hot(
            torch.tensor(int(si.split("_")[-1].replace(".png", ""))),
            num_classes=num_classes,
        ).float()
        for si in os.listdir(slide_dir)
    }

    # Extract patient IDs from the file names
    # Assuming patient_id is the first part before underscore
    patients = np.unique([si.split("_")[0] for si in os.listdir(slide_dir)])

    # Prepare a DataFrame to hold the data
    data = pd.DataFrame()
    data["patient_id"] = [si.split("_")[0] for si in os.listdir(slide_dir)]
    data["filename"] = [si for si in os.listdir(slide_dir)]
    data["block_id"] = [si.split("-")[0] for si in os.listdir(slide_dir)]
    data["label"] = [
        torch.nn.functional.one_hot(
            torch.tensor(int(si.split("_")[-1].replace(".png", ""))),
            num_classes=num_classes,
        ).float()
        for si in os.listdir(slide_dir)
    ]

    # Split the data at the patient level using GroupShuffleSplit
    test_size = config["split"]["test_size"]
    random_seed_train_test = config["split"]["random_seed_train_test"]

    splitter = GroupShuffleSplit(
        test_size=test_size, n_splits=1, random_state=random_seed_train_test
    )
    train_ind, test_ind = tuple(
        splitter.split(X=data.index, groups=data["patient_id"])
    )[0]
    train_table = data.iloc[train_ind]
    test_table = data.iloc[test_ind]
    assert (
        len(set(train_table["patient_id"]).intersection(set(test_table["patient_id"])))
        == 0
    )

    # Further split the test set into validation and test
    val_test_split = config["split"]["val_test_split"]
    random_seed_val_test = config["split"]["random_seed_val_test"]

    splitter = GroupShuffleSplit(
        test_size=val_test_split, n_splits=1, random_state=random_seed_val_test
    )
    val_ind, test_ind = tuple(
        splitter.split(
            X=list(range(0, test_table.shape[0])), groups=test_table["patient_id"]
        )
    )[0]
    val_table = test_table.iloc[val_ind]
    test_table = test_table.iloc[test_ind]
    assert (
        len(set(val_table["patient_id"]).intersection(set(test_table["patient_id"])))
        == 0
    )

    # Generate the labels_dict for training, validation, and testing based on slides
    train_labels_dict = {
        block_id: labels_dict[block_id] for block_id in train_table["block_id"]
    }
    val_labels_dict = {
        block_id: labels_dict[block_id] for block_id in val_table["block_id"]
    }
    test_labels_dict = {
        block_id: labels_dict[block_id] for block_id in test_table["block_id"]
    }

    # Print out the number of unique slides in each set
    print("\nNumber of unique slides in each set:")
    print(f"Train: {len(train_labels_dict)}")
    print(f"Validation: {len(val_labels_dict)}")
    print(f"Test: {len(test_labels_dict)}")

    return train_labels_dict, val_labels_dict, test_labels_dict
