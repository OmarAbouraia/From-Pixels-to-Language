
import torch
from src.data.data_loader_v3 import create_v3_data_loaders

print("Loading CUB-200 dataset...")

train_loader, val_loader = create_v3_data_loaders(
    batch_size=32,
    num_workers=0,
    augmentation="strong"
)

print("\n--- Dataset Information ---")
print("Training images:", len(train_loader.dataset))
print("Validation images:", len(val_loader.dataset))

# Get one training batch
train_batch = next(iter(train_loader))

print("\n--- Training Batch ---")
print("Image shape:", train_batch["image"].shape)
print("Label shape:", train_batch["label"].shape)
print("Image dtype:", train_batch["image"].dtype)
print("Label dtype:", train_batch["label"].dtype)

print(
    "Image value range:",
    train_batch["image"].min().item(),
    train_batch["image"].max().item()
)

# Get one validation batch
val_batch = next(iter(val_loader))

print("\n--- Validation Batch ---")
print("Image shape:", val_batch["image"].shape)
print("Label shape:", val_batch["label"].shape)

# Verify dimensions and labels
assert len(train_loader.dataset) == 4795
assert len(val_loader.dataset) == 1198

assert train_batch["image"].shape == (32, 3, 224, 224)
assert val_batch["image"].shape == (32, 3, 224, 224)

assert train_batch["image"].dtype == torch.float32
assert train_batch["label"].dtype == torch.long

assert torch.isfinite(train_batch["image"]).all()
assert torch.isfinite(val_batch["image"]).all()

assert train_batch["label"].min() >= 0
assert train_batch["label"].max() < 200

print("\nSUCCESS! CNN V3 DataLoader works correctly!")
