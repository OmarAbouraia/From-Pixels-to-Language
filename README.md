# From-Pixels-to-Language
Deep Learning project: image classification, colorization, and captioning

## Data pipeline

Install the project dependencies from the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

The reusable loader reads the checked-in train/validation indices from
`outputs/eda/split_indices.json`, loads `Multimodal-Fatima/CUB_train` once, and
does not load or use the official test dataset:

```python
from src.data.data_loader import create_data_loaders

train_loader, val_loader = create_data_loaders(batch_size=32)
batch = next(iter(train_loader))

print(batch["image"].shape)       # [B, 3, 224, 224], float32, values in [0, 1]
print(batch["label"].shape)       # [B], torch.long
print(batch["captions"][0][0])    # first caption for the first image
```

Run this example from the repository root. `batch_size` defaults to 32; the
final batch may be smaller. Captions are returned as lists of strings (one
list per image) and are not tokenized. The shared split contains 4,795 training
and 1,199 validation examples. Its file is validated for required keys,
duplicate or overlapping indices, and indices outside the loaded dataset.

Images are converted to RGB, resized to 224 × 224, and converted to tensors.
No normalization or random augmentation is applied here. Choose any
model-specific normalization or pretrained-weight preprocessing in coordination
with the selected model; do not silently change the shared transform.
