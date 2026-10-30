"""Reusable DataLoaders for the CUB training and validation splits."""

import json
from pathlib import Path
from typing import TypedDict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from .dataset import CUBDataset
from .transforms import get_image_transforms


DATASET_NAME = "Multimodal-Fatima/CUB_train"


class CUBSample(TypedDict):
    image: torch.Tensor
    label: torch.Tensor
    captions: list[str]


class CUBBatch(TypedDict):
    image: torch.Tensor
    label: torch.Tensor
    captions: list[list[str]]


def collate_cub_batch(batch: list[CUBSample]) -> CUBBatch:
    """Stack image and label tensors while keeping captions as nested lists."""
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "captions": [item["captions"] for item in batch],
    }


def _read_split_indices(split_path: Path) -> tuple[list[int], list[int]]:
    try:
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Split indices file was not found: {split_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Split indices file is not valid JSON: {split_path}"
        ) from error

    if not isinstance(split_data, dict):
        raise ValueError(f"Split indices file must contain a JSON object: {split_path}")

    required_keys = {"train_indices", "val_indices"}
    missing_keys = required_keys.difference(split_data)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Split indices file is missing required key(s): {missing}")

    indices_by_split = {}
    for split_name, key in (
        ("training", "train_indices"),
        ("validation", "val_indices"),
    ):
        indices = split_data[key]
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"'{key}' must be a non-empty list of integer indices.")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise ValueError(f"'{key}' must contain only integer indices.")
        if len(indices) != len(set(indices)):
            raise ValueError(f"'{key}' contains duplicate indices.")
        indices_by_split[split_name] = indices

    overlap = set(indices_by_split["training"]).intersection(
        indices_by_split["validation"]
    )
    if overlap:
        example = min(overlap)
        raise ValueError(
            f"Training and validation indices overlap (for example: {example})."
        )

    return indices_by_split["training"], indices_by_split["validation"]


def _validate_index_range(
    indices: list[int], split_name: str, dataset_length: int
) -> None:
    invalid_indices = [
        index for index in indices if index < 0 or index >= dataset_length
    ]
    if invalid_indices:
        raise ValueError(
            f"{split_name} index {invalid_indices[0]} is outside the loaded "
            f"training dataset range [0, {dataset_length - 1}]."
        )


def create_data_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    split_file: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Load CUB training data and return its training and validation DataLoaders.

    Relative split-file paths are resolved from the project root. The official
    test dataset is intentionally not loaded by this function.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    project_root = Path(__file__).resolve().parents[2]
    if split_file is None:
        split_path = project_root / "outputs" / "eda" / "split_indices.json"
    else:
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = project_root / split_path

    train_indices, val_indices = _read_split_indices(split_path)

    # Load the shared training dataset once for both subsets.
    training_data = load_dataset(DATASET_NAME, split="train")
    dataset_length = len(training_data)
    _validate_index_range(train_indices, "Training", dataset_length)
    _validate_index_range(val_indices, "Validation", dataset_length)

    train_transform = get_image_transforms(training=True)
    val_transform = get_image_transforms(training=False)

    train_dataset = CUBDataset(
        training_data,
        train_indices,
        transform=train_transform,
    )

    val_dataset = CUBDataset(
        training_data,
        val_indices,
        transform=val_transform,
    )

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