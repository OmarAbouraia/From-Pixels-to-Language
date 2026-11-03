
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# Fixed RGB normalization values
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_v3_transforms(training=False, preset="strong"):
    """
    Preprocessing for CNN V3.

    Training:
        Augmentation + normalization

    Validation:
        Deterministic preprocessing + normalization

    Input:
        PIL RGB image

    Output:
        Tensor [3, 224, 224]
    """

    if training:

        if preset == "strong":
            return transforms.Compose([
                # Random crop while preserving most of the image
                transforms.RandomResizedCrop(
                    size=224,
                    scale=(0.80, 1.0),
                    ratio=(0.85, 1.15),
                    interpolation=InterpolationMode.BILINEAR
                ),

                # Randomly flip horizontally
                transforms.RandomHorizontalFlip(p=0.5),

                # Slight rotation
                transforms.RandomRotation(degrees=10),

                # Moderate color and lighting changes
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                    hue=0.03
                ),

                # Convert PIL image to tensor
                transforms.ToTensor(),

                # Normalize RGB channels
                transforms.Normalize(
                    mean=MEAN,
                    std=STD
                )
            ])

        elif preset == "basic":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])

        else:
            raise ValueError(
                f"Unknown augmentation preset: {preset}"
            )

    # Validation: no random augmentation
    return transforms.Compose([
        transforms.Resize(
            256,
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
