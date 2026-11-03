
import torch
from torch.utils.data import Dataset


class CUBDataset(Dataset):
    def __init__(self, huggingface_dataset, indices, transform=None):
        self.dataset = huggingface_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # Get the original dataset index for this example
        original_index = self.indices[index]
        example = self.dataset[original_index]

        # Load the image, species label, and captions
        image = example["image"].convert("RGB")
        label = example["label"]
        captions = example["description"].splitlines()

        # Apply image preprocessing
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "captions": captions,
        }
