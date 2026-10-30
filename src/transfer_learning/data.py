import json
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader

from src.data.dataset import CUBDataset
from src.data.data_loader import DATASET_NAME, collate_cub_batch

from .transforms import get_transfer_learning_transforms


def load_shared_split(split_file=None):
    """
    Load and validate the existing EDA train/validation split.

    No new split is generated.
    """

    project_root = Path(__file__).resolve().parents[2]

    if split_file is None:
        split_path = (
            project_root / "outputs" / "eda" / "split_indices.json"
        )
    else:
        split_path = Path(split_file)

        if not split_path.is_absolute():
            split_path = project_root / split_path

    with split_path.open("r", encoding="utf-8") as file:
        split_data = json.load(file)

    if not isinstance(split_data, dict):
        raise ValueError("Split file must contain a JSON object.")

    train_indices = split_data.get("train_indices")
    val_indices = split_data.get("val_indices")

    for name, indices in (
        ("train_indices", train_indices),
        ("val_indices", val_indices),
    ):
        if (
            not isinstance(indices, list)
            or not indices
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in indices
            )
        ):
            raise ValueError(f"Invalid {name} in split file.")

        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate indices found in {name}.")

    if set(train_indices).intersection(val_indices):
        raise ValueError("Training and validation indices overlap.")

    return train_indices, val_indices


def create_transfer_learning_loaders(
    batch_size=32,
    num_workers=0,
    split_file=None,
):
    """
    Create Transfer Learning training and validation DataLoaders.

    Uses:
        - Existing CUBDataset implementation
        - Existing EDA train/validation indices
        - ImageNet-normalized transformations
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    # Load the existing EDA split
    train_indices, val_indices = load_shared_split(split_file)

    # Load the Hugging Face dataset once
    cub_dataset = load_dataset(DATASET_NAME, split="train")

    dataset_length = len(cub_dataset)

    # Validate indices against the loaded dataset
    for split_name, indices in (
        ("Training", train_indices),
        ("Validation", val_indices),
    ):
        invalid_indices = [
            index for index in indices
            if index < 0 or index >= dataset_length
        ]

        if invalid_indices:
            raise ValueError(
                f"{split_name} index {invalid_indices[0]} "
                f"is outside the dataset range."
            )

    # Create separate transformations
    train_transform = get_transfer_learning_transforms(training=True)
    val_transform = get_transfer_learning_transforms(training=False)

    # Reuse the existing CUBDataset class
    train_dataset = CUBDataset(
        cub_dataset,
        train_indices,
        transform=train_transform,
    )

    val_dataset = CUBDataset(
        cub_dataset,
        val_indices,
        transform=val_transform,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_cub_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_cub_batch,
    )

    return train_loader, val_loader