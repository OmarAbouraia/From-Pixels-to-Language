
"""DataLoaders for CNN V3 using V3-specific preprocessing."""

from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader

from .dataset import CUBDataset

from .data_loader import (
    DATASET_NAME,
    _read_split_indices,
    _validate_index_range,
    collate_cub_batch,
)

from .transforms_v3 import get_v3_transforms


def create_v3_data_loaders(
    batch_size=32,
    num_workers=0,
    augmentation="strong",
    split_file=None,
):
    """
    Create CNN V3 training and validation DataLoaders.

    Uses the same dataset and split indices as V1/V2.

    Only preprocessing changes for V3.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    # Locate the project root
    project_root = Path(__file__).resolve().parents[2]

    # Use the existing saved split
    if split_file is None:
        split_path = (
            project_root
            / "outputs"
            / "eda"
            / "split_indices.json"
        )
    else:
        split_path = Path(split_file)

        if not split_path.is_absolute():
            split_path = project_root / split_path

    # Reuse teammate's split validation
    train_indices, val_indices = _read_split_indices(
        split_path
    )

    # Load the CUB training source once
    training_data = load_dataset(
        DATASET_NAME,
        split="train"
    )

    # Validate index ranges
    dataset_length = len(training_data)

    _validate_index_range(
        train_indices,
        "Training",
        dataset_length
    )

    _validate_index_range(
        val_indices,
        "Validation",
        dataset_length
    )

    # V3-specific image preprocessing
    train_transform = get_v3_transforms(
        training=True,
        preset=augmentation
    )

    val_transform = get_v3_transforms(
        training=False
    )

    # Use the same CUBDataset class
    train_dataset = CUBDataset(
        training_data,
        train_indices,
        transform=train_transform
    )

    val_dataset = CUBDataset(
        training_data,
        val_indices,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_cub_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_cub_batch
    )

    return train_loader, val_loader
