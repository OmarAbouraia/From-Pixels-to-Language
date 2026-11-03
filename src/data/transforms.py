
from torchvision import transforms


def get_image_transforms(training=False):
    """
    Preprocess images for training or validation.

    Training images receive random augmentation.
    Validation images receive only consistent preprocessing.
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
    ])

    return transforms.Compose(image_transforms)
