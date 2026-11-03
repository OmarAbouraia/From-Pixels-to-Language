from torchvision import transforms


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transfer_learning_transforms(training=False):
    """
    Image preprocessing for pretrained ImageNet models.

    Training:
        - Random horizontal flip
        - Random rotation
        - Resize
        - Convert to tensor
        - ImageNet normalization

    Validation:
        - Resize
        - Convert to tensor
        - ImageNet normalization
    """

    image_transforms = []

    if training:
        image_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])

    image_transforms.extend([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])

    return transforms.Compose(image_transforms)