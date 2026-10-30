# From-Pixels-to-Language

Deep Learning project: image classification, colorization, and captioning.

## 1. Project Overview

This project uses the CUB-200-2011 bird dataset for image classification, image colorization, and image captioning. This README section documents **Workstream A: Exploratory Data Analysis (EDA) and the shared data pipeline**.

## 2. Workstream A Scope

The EDA notebook inspects the Hugging Face dataset, class and image distributions, caption structure, duplicate images, and basic data integrity. The reusable Python data pipeline is in `src/data/`; it loads the training source, applies shared image preprocessing, and returns batches for the classification and later multimodal tasks.

The EDA results inform downstream work: the classes are close to balanced, original image sizes vary, the captions have a broad length range, and one exact duplicate pair needed to be kept out of separate train/validation splits.

## 3. Dataset Overview

The notebook loads these two Hugging Face dataset configurations:

| Source                                          | Use                     | Images |
| ----------------------------------------------- | ----------------------- | -----: |
| `Multimodal-Fatima/CUB_train` (`split="train"`) | Training and validation |  5,994 |
| `Multimodal-Fatima/CUB_test` (`split="test"`)   | Official test set       |  5,794 |

The notebook reports 200 bird species. Each record includes an image, a class label, a filename, and a `description` field containing ten human-written captions separated by line breaks. The reusable loader uses only `CUB_train`; it does not load the official test set. Keep the official test data for final evaluation rather than model selection.

## 4. EDA Findings

Findings below are from the saved outputs in `src/data/01_dataset_inspection.ipynb`:

| Area               | Finding                                                                                                                                                        | Why it matters                                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Class distribution | All 200 classes are represented. Training counts range from 29 to 30 images per class (mean 29.97). The official test counts range from 11 to 30 (mean 28.97). | The training set is nearly balanced, but test class counts vary; report class-aware metrics as well as overall accuracy.    |
| Image dimensions   | There are 839 unique original width-height pairs.                                                                                                              | A fixed model input size is needed; resizing every image to a square can change its aspect ratio.                           |
| Caption count      | Every training image has 10 captions, for 59,940 captions total.                                                                                               | The dataset provides multiple references per image for captioning.                                                          |
| Caption length     | Mean 15.25 words; minimum 1 word; maximum 65 words. The notebook lowercases text and counts word-like tokens for this statistic.                               | Caption batches may need padding/truncation when a captioning tokenizer is added; choose limits based on this distribution. |
| Caption vocabulary | Approximately 4,713 unique lowercased tokens under the notebook's word-counting rule.                                                                          | Vocabulary and tokenization choices should be made for the captioning model, not assumed by the image loader.               |

The short-caption inspection highlights `"this"` and `"the colorful bird"`. These may be low-information captions, but they are not automatically invalid and are not deleted by the data pipeline.

## 5. Data-Quality Checks

* **Image decoding:** all 5,994 training images were decoded; zero loading failures were recorded.
* **Empty captions:** all 59,940 training captions were checked; zero were empty after whitespace checks.
* **Exact duplicates:** the notebook hashes decoded RGB pixel bytes and found one exact duplicate pair, original dataset indices 1712 and 1731. Both have label 57, which the dataset maps to **Pigeon Guillemot**. The notebook image filenames also identify both as Pigeon Guillemots. The class name is not Olive-sided Flycatcher.
* **Duplicate handling:** index 1712 was removed from validation; index 1731 remains in training. This prevents the same image content appearing on both sides of the train/validation boundary. The duplicate check detects exact pixel matches, not visually similar near-duplicates.
* **Caption quality:** the notebook flags unusually short examples but does not detect or correct typos, and the loader does not clean or tokenize caption text.

## 6. Preprocessing and Augmentation

`src/data/dataset.py` and `src/data/transforms.py` implement these steps:

1. Convert each image to RGB so every example has three color channels.
2. During training only, randomly flip horizontally and rotate by up to 10 degrees. These modest augmentations add visual variation while retaining bird identity and label.
3. Resize images to 224 × 224 and convert them to PyTorch tensors with values in `[0, 1]`. Resizing to a fixed square shape gives the models a consistent input shape, but may distort the original aspect ratio.
4. Split `description` on line breaks into a list of captions. No caption cleaning, correction, or tokenization is performed in the loader.
5. Use deterministic RGB, resize, and tensor conversion for validation; random augmentation is not applied there.
6. Collate images and labels by stacking tensors, while preserving captions as nested lists (one list of strings per image).

`create_data_loaders()` defaults to a batch size of 32 and zero worker processes. The final batch may be smaller. No pixel normalization is applied; any model-specific normalization should be explicit and consistent with that model's requirements.

## 7. Train/Validation Split

The notebook creates a stratified 80/20 split from the 5,994-image training dataset with `random_state=42`, then removes duplicate index 1712 from validation. The checked-in `outputs/eda/split_indices.json` therefore contains:

* Training: 4,795 indices
* Validation: 1,198 indices
* Total included: 5,993 (one duplicate record excluded)
* Train/validation overlap: none
* Classes represented: all 200

The earlier 4,795/1,199 count was the split before removing the duplicate and is no longer the checked-in split. `src/data/data_loader.py` validates the split file's required keys, unique indices, no overlap, and index ranges before building loaders.

## 8. EDA Plots

The notebook contains code to save these plots under `outputs/eda/`:

| Plot                              | What it shows                                            |
| --------------------------------- | -------------------------------------------------------- |
| `train_class_distribution.png`    | Training image counts across the 200 classes.            |
| `sample_images_and_sizes.png`     | Example training images with their original dimensions.  |
| `sample_images_and_captions.png`  | Example images with a reference caption and class label. |
| `test_class_distribution.png`     | Official test image counts by class.                     |
| `duplicate_images.png`            | The duplicate pair, with indices, filenames, and labels. |
| `caption_length_distribution.png` | Caption word-count distribution.                         |
| `image_dimensions.png`            | Scatter plot of original image width versus height.      |
| `top_caption_words.png`           | The 20 most frequent lowercased caption words.           |

**Artifact status:** The eight EDA plot files and the final `split_indices.json` are stored in the repository's `outputs/eda/` directory.

## 9. Project Structure

```text
src/
└── data/
    ├── 01_dataset_inspection.ipynb  # EDA and exploratory plots
    ├── data_loader.py               # train/validation DataLoaders and split checks
    ├── dataset.py                   # RGB conversion and caption line splitting
    └── transforms.py                # train/validation image transforms

outputs/
└── eda/
    ├── caption_length_distribution.png
    ├── duplicate_images.png
    ├── image_dimensions.png
    ├── sample_images_and_captions.png
    ├── sample_images_and_sizes.png
    ├── split_indices.json
    ├── test_class_distribution.png
    ├── top_caption_words.png
    └── train_class_distribution.png

tests/
└── test_data_loader.py              # data-loader unit tests
```

## 10. Run the Notebook and Tests

Run commands from the repository root. Create and activate the environment, then install the listed project dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open `src/data/01_dataset_inspection.ipynb` in VS Code and select the project Python environment as its kernel. The notebook downloads the public Hugging Face datasets, so a network connection is required. `requirements.txt` includes the data, plotting, and ML packages used by the notebook, but not Jupyter itself; use the VS Code Jupyter extension or an environment with a Jupyter kernel available.

The notebook uses the repository root to locate the centralized `outputs/eda/` directory. The checked-in `split_indices.json` contains the final train/validation split, with duplicate index 1712 excluded from validation. Avoid rerunning split-generation cells unless you intentionally want to regenerate the split and have verified the resulting indices.

Run the unit tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The latest verified run in `.venv` passed all four tests. They check the shared split and data-loader batch contract (image/label shapes and types, pixel range, caption lists, and final partial batch), and reject missing split keys, overlapping indices, and out-of-range indices. The tests mock the Hugging Face dataset; they do not rerun the full EDA audit or test augmentation behavior.

Use the reusable loader from Python:

```python
from src.data.data_loader import create_data_loaders

train_loader, val_loader = create_data_loaders(batch_size=32)
batch = next(iter(train_loader))

print(batch["image"].shape)       # [B, 3, 224, 224], float32, values in [0, 1]
print(batch["label"].shape)       # [B], torch.long
print(batch["captions"][0][0])    # first caption for the first image
```

## 11. Limitations and Notes

* Image quality was reviewed through image samples and decode checks; no objective blur, resolution-quality, or exposure metric is calculated.
* Exact duplicate hashing does not detect near-duplicates.
* The loader only splits captions on line breaks. Caption normalization, vocabulary construction, tokenization, and padding belong to the captioning pipeline.


