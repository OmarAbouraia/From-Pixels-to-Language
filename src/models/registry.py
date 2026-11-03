"""Shared classifier interface for every classification model in the project.

Every model registered here follows the agreed contract
(see docs/interface_contracts.md):

    input : float tensor [B, 3, 224, 224], RGB, NCHW
    output: raw logits   [B, 200]  (no softmax)

The rest of the project (CLI, prediction, Grad-CAM, evaluation) talks to
models ONLY through the functions in this file, so a new encoder (for example
the scratch CNN) can be added by registering it, without touching the
training loop, the CLI or Grad-CAM.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from src.transfer_learning.model import NUM_CLASSES, create_resnet18


@dataclass(frozen=True)
class ModelSpec:
    """Everything the project needs to know about one architecture.

    builder:            function(num_classes, mode, pretrained) -> nn.Module
    gradcam_layer:      function(model) -> the conv layer Grad-CAM hooks into
    normalization:      "imagenet" or "scratch"; decides which transforms
                        prediction uses, so inference matches training.
    """

    builder: Callable[..., nn.Module]
    gradcam_layer: Callable[[nn.Module], nn.Module]
    normalization: str


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(name: str, spec: ModelSpec) -> None:
    """Add an architecture to the registry under a unique name."""
    if name in _REGISTRY:
        raise ValueError(f"Model '{name}' is already registered.")
    _REGISTRY[name] = spec


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{name}'. Available: {', '.join(available_models())}"
        ) from None


def build_classifier(
    name: str,
    num_classes: int = NUM_CLASSES,
    mode: str = "fine_tuning",
    pretrained: bool = True,
) -> nn.Module:
    """Create an (untrained or ImageNet-initialised) classifier by name."""
    return get_model_spec(name).builder(
        num_classes=num_classes, mode=mode, pretrained=pretrained
    )


def get_gradcam_target_layer(name: str, model: nn.Module) -> nn.Module:
    """Return the last convolutional block of the model, used by Grad-CAM."""
    return get_model_spec(name).gradcam_layer(model)


def load_classifier(
    checkpoint_path: str | Path,
    name: str = "resnet18",
    num_classes: int = NUM_CLASSES,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict]:
    """Load a trained checkpoint and return (model in eval mode, metadata).

    Works with the checkpoints saved by src/transfer_learning/train.py
    (a dict containing 'model_state_dict', 'mode', 'learning_rate', ...)
    and also with a bare state_dict.

    pretrained=False is used when building the skeleton: the trained weights
    come from the checkpoint, so downloading ImageNet weights is unnecessary.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.items()
                    if k not in ("model_state_dict", "optimizer_state_dict")}
    else:
        state_dict = checkpoint
        metadata = {}

    model = build_classifier(
        name,
        num_classes=num_classes,
        mode=metadata.get("mode", "fine_tuning"),
        pretrained=False,
    )
    # strict=True: fail loudly if the checkpoint does not match the architecture
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    metadata["model_name"] = name
    metadata["checkpoint_path"] = str(checkpoint_path)
    return model, metadata


# ---------------------------------------------------------------------------
# Registered architectures
# ---------------------------------------------------------------------------

register_model(
    "resnet18",
    ModelSpec(
        builder=create_resnet18,          # Omar's builder, reused unchanged
        gradcam_layer=lambda model: model.layer4[-1],
        normalization="imagenet",
    ),
)

# The scratch CNN is registered here once it is implemented, e.g.:
#
# register_model(
#     "scratch_cnn",
#     ModelSpec(
#         builder=create_scratch_cnn,
#         gradcam_layer=lambda model: model.<last_conv_block>,
#         normalization="scratch",
#     ),
# )
