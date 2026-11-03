"""Predict the bird species of a single image with any registered classifier."""

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from src.data.transforms import get_image_transforms
from src.models import get_model_spec
from src.transfer_learning.transforms import get_transfer_learning_transforms


DATASET_NAME = "Multimodal-Fatima/CUB_train"


def get_eval_transform(model_name: str):
    """Deterministic preprocessing that matches how the model was trained.

    Pretrained ResNet18 was trained with ImageNet normalisation; the scratch
    CNN uses the shared EDA transforms (no normalisation). Using the wrong one
    silently ruins predictions, so it is chosen from the model registry.
    """
    normalization = get_model_spec(model_name).normalization
    if normalization == "imagenet":
        return get_transfer_learning_transforms(training=False)
    return get_image_transforms(training=False)


def load_class_names(class_names_file: str | Path | None = None,
                     num_classes: int = 200) -> list[str]:
    """Return the list of species names, index i = class i.

    Order of preference:
      1. A JSON file containing a list of names (fast, works offline).
      2. The label names stored in the Hugging Face dataset metadata
         (downloads only metadata, not the images).
      3. Fallback "class_<i>" so prediction still works without internet.
    """
    if class_names_file is not None:
        names = json.loads(Path(class_names_file).read_text(encoding="utf-8"))
        if len(names) != num_classes:
            raise ValueError(
                f"{class_names_file} has {len(names)} names, expected {num_classes}."
            )
        return names

    try:
        from datasets import load_dataset_builder

        names = load_dataset_builder(DATASET_NAME).info.features["label"].names
        if len(names) == num_classes:
            return list(names)
    except Exception as error:  # no internet, HF down, etc.
        print(f"WARNING: could not read class names from Hugging Face ({error}).")

    return [f"class_{index}" for index in range(num_classes)]


@torch.no_grad()
def predict_tensor(model: nn.Module, image_tensor: torch.Tensor, top_k: int = 5):
    """Return [(class_index, probability), ...] for one preprocessed image.

    image_tensor: [3, H, W] or [1, 3, H, W].
    Softmax is applied here only, because the model returns raw logits.
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    device = next(model.parameters()).device
    model.eval()
    probabilities = torch.softmax(model(image_tensor.to(device)), dim=1)[0]
    top_probs, top_indices = probabilities.topk(min(top_k, probabilities.numel()))
    return [(int(i), float(p)) for i, p in zip(top_indices, top_probs)]


def predict_image(model: nn.Module, model_name: str, image_path: str | Path,
                  class_names: list[str], top_k: int = 5) -> list[dict]:
    """Load an image from disk and return its top-k predicted species."""
    image = Image.open(image_path).convert("RGB")
    tensor = get_eval_transform(model_name)(image)
    return [
        {"rank": rank, "class_index": index, "species": class_names[index],
         "probability": probability}
        for rank, (index, probability) in enumerate(
            predict_tensor(model, tensor, top_k), start=1
        )
    ]
