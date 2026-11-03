import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 200


def create_resnet18(
    num_classes=NUM_CLASSES,
    mode="feature_extraction",
    pretrained=True,
):
    """
    Create a ResNet18 model for CUB-200 bird classification.

    Args:
        num_classes: Number of output classes.
        mode: 'feature_extraction' or 'fine_tuning'.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        Configured ResNet18 model.
    """

    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero.")

    if mode not in ("feature_extraction", "fine_tuning"):
        raise ValueError(
            "mode must be 'feature_extraction' or 'fine_tuning'."
        )

    # Load ResNet18
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Freeze the backbone for feature extraction
    if mode == "feature_extraction":
        for parameter in model.parameters():
            parameter.requires_grad = False

    # Replace the original ImageNet classification head
    input_features = model.fc.in_features

    model.fc = nn.Linear(input_features, num_classes)

    # In fine-tuning mode, all parameters must be trainable
    if mode == "fine_tuning":
        for parameter in model.parameters():
            parameter.requires_grad = True

    return model