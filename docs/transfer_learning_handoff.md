# Transfer Learning Handoff — From Pixels to Language

**Purpose:** Handoff notes for the teammate continuing the transfer-learning work and integrating it into Workstream D.  
**Project:** CSE 485 / CSE 488 — Deep Learning, *From Pixels to Language*  
**Dataset:** CUB-200-2011 with captions  
**Owner of previous transfer-learning implementation:** Omar Abouraia  
**Last known transfer-learning commit:** `99fae0b` — configurable weight decay support

**Current branch status (reported 2 Nov 2026):** `transfer-learning` is up to date with `origin/transfer-learning`. Three untracked root-level test scripts remain locally; they were not included in the handoff commit.  
**Branch:** `transfer-learning`

> Before making changes, check the live GitHub branch and commit history. The commit and status above reflect the last recorded handoff context and may have changed.

---

## 1. Handoff summary

The transfer-learning workstream has implemented and evaluated ResNet18 on the CUB 200-species classification task.

Completed:
- ResNet18 with ImageNet-pretrained weights.
- Feature Extraction: backbone frozen; only the 200-class fully connected layer trained.
- Fine-Tuning: all ResNet18 layers trained.
- Reuse of the project's existing EDA train/validation split.
- Transfer-learning-specific image transforms.
- Training and validation loops.
- CLI for launching training experiments.
- Checkpoint and CSV history saving.
- Evaluation of Feature Extraction, Fine-Tuning, WD-0001, and WD-001.
- Confusion matrices, prediction exports, evaluation metrics, and comparison plots for the completed evaluations.
- Configurable weight decay in the training CLI.

The main result so far is that full Fine-Tuning performed better than Feature Extraction on the validation split.

Not yet completed / needs verification:
- WD-01 checkpoint evaluation and its `evaluation_metrics.json`, confusion matrix, and predictions.
- Formal Grid Search and Random Search, and a comparison of the two methods. The existing weight-decay runs alone do **not** prove that both required search methods have been completed.
- Final integration with the shared encoder interface.
- Classification inference CLI (verify whether another teammate has implemented it).
- Experiment tracking integration (verify whether another teammate has implemented it).
- Grad-CAM (assigned to Workstream D).
- Final reproducibility and README checks.

---

## 2. Repository and code

**GitHub repository:**  
`https://github.com/OmarAbouraia/From-Pixels-to-Language`

**Relevant branch:**  
`transfer-learning`

**Last known relevant commit:**  
`99fae0b` — added configurable weight decay support to `src/transfer_learning/train.py`.

**Local project path used by Omar:**
```text
D:\ASU\Senior\Semester 7\Deep Learning\From-Pixels-to-Language-1
```

### Final Git workflow agreed for this handoff

Use the existing `transfer-learning` branch for the remaining transfer-learning work. Create a separate branch for Workstream D only after the transfer-learning work is ready to merge.

**Step 1 — Omar publishes the current handoff**
- Omar adds the experiment notebook under `notebooks/experiments/` (if it is ready and appropriate for the repository) and this handoff document to `transfer-learning`.
- He commits and pushes those files.
- Do not merge the branch yet, because the transfer-learning tuning work is still outstanding.
- Do not stage the three untracked root-level test scripts automatically; inspect them and decide separately whether they belong in the repository.

Example:
```bash
git switch transfer-learning
git status
git add notebooks/experiments/transfer_learning_experiments.ipynb
git add transfer_learning_handoff.md
git status
git commit -m "Add transfer learning experiment notebook and handoff documentation"
git push origin transfer-learning
```

**Step 2 — The next developer continues transfer-learning on the same branch**
- The next developer pulls the latest `origin/transfer-learning`.
- They complete the remaining transfer-learning tasks on that branch: evaluate the saved WD-01 checkpoint, implement and document Grid Search and Random Search, compare the search methods fairly, and record the selected configuration and results.
- They commit and push their progress to `transfer-learning`.
- Coordinate before simultaneous edits. Do not have two people independently modify the same files without agreement.

Example:
```bash
git fetch origin
git switch transfer-learning
git pull origin transfer-learning
# Continue tuning, evaluation, and comparison work.
git add <reviewed-transfer-learning-files>
git commit -m "Complete transfer learning tuning experiments"
git push origin transfer-learning
```

**Step 3 — Merge transfer-learning, then branch for Workstream D**
- Once the remaining transfer-learning work is complete, create a PR from `transfer-learning` into `main`, review it, and merge it.
- Then create `workstream-d-integration` from the updated `main`. This ensures the integration branch starts with all accepted transfer-learning work and its eventual PR does not redundantly include unmerged transfer-learning commits.
- Implement Workstream D on `workstream-d-integration`: shared model/encoder interface, Grad-CAM, CLI integration, experiment tracking coordination, inference tests, and reproducibility.
- When ready, create a second PR from `workstream-d-integration` into `main`.

Example after the transfer-learning PR is merged:
```bash
git switch main
git pull origin main
git switch -c workstream-d-integration
git push -u origin workstream-d-integration
```

When Workstream D is complete:
```text
base: main
compare: workstream-d-integration
```

**Branch flow:**
```text
main
  └── transfer-learning
        ├── Omar's current implementation and handoff
        └── continued tuning + Grid Search / Random Search
              |
              └── PR to main, then merge
                    |
                    └── workstream-d-integration (created from updated main)
                          └── Workstream D changes
                                └── PR to main
```

Important: the next developer should not create the integration branch from an unmerged `transfer-learning` branch if the team intends to submit a clean, integration-only PR to `main`. Merge transfer-learning first, then branch from updated `main`.

Keep checkpoints and datasets out of Git; they are stored separately in Google Drive. Notebooks are experiment records only. The final training and inference implementation must remain in Python modules and be callable through the CLI.

Useful initial commands:
```bash
git status
git branch -a
git log --oneline --decorate -10
```

---

## 3. Code structure and implementation notes

The transfer-learning implementation is under:
```text
src/transfer_learning/
    data.py
    transforms.py
    model.py
    engine.py
    train.py
```

Related existing project data modules:
```text
src/data/
    dataset.py
    data_loader.py
    transforms.py
```

### Data and split

- Dataset source: Hugging Face `Multimodal-Fatima/CUB_train`.
- Number of classes: 200.
- The existing EDA split is the source of truth:
  - Train: 4,795 images.
  - Validation: 1,198 images.
- Split file: `outputs/eda/split_indices.json`.
- Preserve this split. Do not create a new random split for transfer-learning comparisons.
- The validation loader uses `shuffle=False`, allowing prediction arrays from different models to be compared image by image.
- The dataset loader returns batches as dictionaries:
```python
images = batch["image"].to(device)
labels = batch["label"].to(device)
```
- Labels are integer class indices in the range 0–199.

### Image transforms

- Resize to 224 × 224.
- Convert to tensor.
- Normalize using ImageNet mean and standard deviation.
- Training additionally uses random horizontal flip and random rotation (10 degrees).
- Validation uses deterministic transforms.
- Transfer-learning transforms are separate from the original EDA transforms to avoid changing the EDA pipeline.

### Model

- Architecture: torchvision ResNet18.
- Pretraining: ImageNet.
- Classifier output: 200 classes.
- Feature Extraction: freeze the backbone; train only the new classifier.
- Fine-Tuning: train the full network.
- Classification loss: `CrossEntropyLoss` applied to raw logits (do not apply Softmax before this loss).

### Training

The training entry point is:
```bash
python -m src.transfer_learning.train --help
```

Known CLI options:
```text
--mode {feature_extraction,fine_tuning}
--epochs
--batch-size
--learning-rate
--weight-decay
--num-workers
--output-dir
```

Typical settings used:
- Batch size: 32.
- Feature Extraction learning rate: 0.001.
- Fine-Tuning learning rate: 0.0001.
- Optimizer: Adam.
- Weight decay: configurable; default 0.0.
- Epochs: 10 for the reported experiments.

The training code saves:
- `best_model.pth`
- `last_checkpoint.pth`
- `training_history.csv`

The best checkpoint is selected by minimum validation loss.

**Checkpoint caution:** The current training pipeline was not known to have a complete resume-from-checkpoint CLI. Do not assume that `last_checkpoint.pth` can be used to resume training automatically. Check the checkpoint structure and training code first. For inference, load the model architecture, load `model_state_dict`, and call `eval()`.

---

## 4. Existing experiment results

All results below use the same 4,795 / 1,198 train/validation split.

### Feature Extraction

Configuration:
- ResNet18, ImageNet pretrained.
- Frozen backbone; train classifier only.
- 10 epochs.
- Batch size 32.
- Adam, learning rate 0.001.

Best validation checkpoint:
- Best epoch: 10.
- Validation loss: 1.6689.
- Validation accuracy: 54.76%.

Evaluation:
| Metric | Result |
|---|---:|
| Accuracy | 54.76% |
| Top-5 accuracy | 84.81% |
| Macro precision | 58.73% |
| Macro recall | 54.73% |
| Macro F1 | 53.92% |
| Weighted precision | 58.75% |
| Weighted recall | 54.76% |
| Weighted F1 | 53.94% |

### Fine-Tuning baseline

Configuration:
- ResNet18, ImageNet pretrained.
- All layers trainable.
- 10 epochs.
- Batch size 32.
- Adam, learning rate 0.0001.
- Weight decay 0.0.

Best validation checkpoint:
- Best epoch: 10.
- Validation loss: 1.3116.
- Validation accuracy: 67.70%.

Evaluation:
| Metric | Result |
|---|---:|
| Accuracy | 67.70% |
| Top-5 accuracy | 90.90% |
| Macro precision | 70.87% |
| Macro recall | 67.68% |
| Macro F1 | 67.48% |
| Weighted precision | 70.89% |
| Weighted recall | 67.70% |
| Weighted F1 | 67.49% |

### Weight-decay experiments

All three used Fine-Tuning, 10 epochs, batch size 32, learning rate 0.0001, and seed 42.

| Run | Weight decay | Best validation loss | Validation accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Fine-Tuning baseline | 0 | 1.3116 | 67.70% | 67.48% |
| WD-0001 | 0.0001 | 1.3422 | 65.86% | 65.91% |
| WD-001 | 0.001 | 1.3405 | 66.69% | 66.83% |
| WD-01 | 0.01 | 1.5584 | 63.36% | **Not evaluated yet** |

WD-01's training history and checkpoints are saved, but its final evaluation artifacts were not present in the last Drive inventory. Evaluate this existing checkpoint; do not retrain it merely to obtain evaluation metrics.

### Main comparison

| Metric | Feature Extraction | Fine-Tuning |
|---|---:|---:|
| Validation loss | 1.6689 | **1.3116** |
| Accuracy | 54.76% | **67.70%** |
| Top-5 accuracy | 84.81% | **90.90%** |
| Macro precision | 58.73% | **70.87%** |
| Macro recall | 54.73% | **67.68%** |
| Macro F1 | 53.92% | **67.48%** |

Fine-Tuning is currently the best-performing transfer-learning configuration among the completed experiments.

### Additional analysis already performed

- Training/validation loss and accuracy curves.
- Confusion matrix analysis.
- Per-class precision, recall, and F1 comparison.
- Image-level paired comparison:
  - Both models correct: 574.
  - Feature Extraction correct, Fine-Tuning wrong: 82.
  - Feature Extraction wrong, Fine-Tuning correct: 237.
  - Both wrong: 305.
- Fine-Tuning corrected 237 Feature Extraction errors and introduced 82 new errors, for a net gain of 155 correctly classified validation images.

Remember that each validation class has only six images, so per-class metrics are noisy and should be interpreted cautiously.

---

## 5. Google Drive experiment artifacts

**Google Drive root folder:**
```text
MyDrive/From-Pixels-to-Language/Transfer-Learning/
```

**Shareable Drive folder link:**  
`https://drive.google.com/drive/folders/1fEdPBCe4d1G1iotcUyqYxE3Fdera2z0G?usp=drive_link`

The last verified inventory contained:

```text
Transfer-Learning/
├── Feature-Extraction/
│   ├── best_model.pth
│   ├── last_checkpoint.pth
│   ├── training_history.csv
│   ├── evaluation_metrics.json
│   ├── confusion_matrix.csv
│   └── validation_predictions.csv
│
├── Fine-Tuning/
│   ├── best_model.pth
│   ├── last_checkpoint.pth
│   ├── training_history.csv
│   ├── evaluation_metrics.json
│   ├── confusion_matrix.csv
│   ├── validation_predictions.csv
│   └── Plots/
│       ├── training_accuracy_comparison.png
│       ├── training_loss_comparison.png
│       ├── validation_accuracy_comparison.png
│       └── validation_loss_comparison.png
│
└── Hyperparameter-Tuning/
    ├── WD-0001/
    │   ├── best_model.pth
    │   ├── last_checkpoint.pth
    │   ├── training_history.csv
    │   ├── evaluation_metrics.json
    │   ├── confusion_matrix.csv
    │   └── validation_predictions.csv
    │
    ├── WD-001/
    │   ├── best_model.pth
    │   ├── last_checkpoint.pth
    │   ├── training_history.csv
    │   ├── evaluation_metrics.json
    │   ├── confusion_matrix.csv
    │   └── validation_predictions.csv
    │
    └── WD-01/
        ├── best_model.pth
        ├── last_checkpoint.pth
        └── training_history.csv
```

The Drive inventory showed approximately 43.89 MB per Feature Extraction checkpoint and 129.21 MB per Fine-Tuning checkpoint.

### How the next developer should use these files

- Use `best_model.pth` for inference and model comparison.
- Use `training_history.csv` for plots and experiment summaries.
- Use evaluation JSON/CSV files where available.
- Evaluate WD-01 from its saved `best_model.pth`.
- Do not retrain completed experiments unless there is a clear, agreed reason.
- Do not assume `last_checkpoint.pth` is resumable without checking the checkpoint contents and training implementation.
- Keep the Drive folder shared with sufficient permissions for the teammate. Prefer Editor access if they need to add evaluation outputs or new experiment artifacts.

---

## 6. Notebook handoff

The Colab notebook used to run the experiments should be saved as an `.ipynb` file and shared with the next developer.

**Notebook link:**  
`https://colab.research.google.com/drive/1yttlGryGzjoZSTaBNRu1DIeSzv-dx7H5?usp=drive_link`

Recommended:
1. Save the notebook in Google Drive.
2. Download an `.ipynb` copy as a backup.
3. Add the notebook to the repository under `notebooks/experiments/` if it contains useful, non-sensitive experiment procedures and is not excessively large.
4. Keep the notebook as an experiment record, not as the final training/inference implementation.
5. Remove or replace any personal paths and ensure it mounts the shared Drive folder using a configurable root path.

The teammate should inspect the notebook to understand how datasets were loaded, how experiments were launched, and how checkpoints were evaluated. They should not blindly rerun all cells from the beginning.

---

## 7. Workstream D: integration tasks for the next developer

Workstream D's responsibility is to make the models usable, explainable, and reproducible.

### A. Agree on a shared encoder interface

Coordinate with Hossam and Omar before changing model signatures.

The interface should make it clear:
- How to instantiate the encoder.
- How to select scratch-trained or pretrained weights.
- What tensor shape the encoder accepts.
- Whether it returns logits, feature maps, or both.
- How checkpoints are loaded.
- How the classification head is accessed for Grad-CAM.

Avoid breaking the current transfer-learning model while integrating it.

### B. Grad-CAM

Implement Grad-CAM for:
- Correct classification predictions.
- Incorrect classification predictions.

Use the final convolutional layer of ResNet18 as the target layer (commonly `layer4[-1]`), and verify the hook/output behavior. Save overlays and record the true and predicted labels. Coordinate with the scratch CNN owner because its target layer may differ.

### C. Classification CLI

Build or integrate CLI commands for:
- Training the scratch baseline.
- Training the pretrained model.
- Predicting the class of an input image using either model/checkpoint.

The existing transfer-learning training CLI is:
```bash
python -m src.transfer_learning.train --help
```

Do not replace it without first checking whether the shared CLI can call the existing training function.

### D. Experiment tracking

Coordinate with Omar and Hossam to avoid duplicate tracking systems. At minimum, each run should record:
- Model type and mode.
- Pretrained weights setting.
- Learning rate, weight decay, batch size, epochs, optimizer, seed.
- Train/validation loss and metrics.
- Runtime/duration.
- Checkpoint path or reference.
- Training/validation curves.

### E. Configuration and dependencies

- Provide configuration templates for scratch and pretrained classification.
- Maintain a reproducible `requirements.txt`.
- Avoid changing dependency versions without checking compatibility with the existing code and Colab environment.

### F. Inference validation

Verify that both the scratch baseline and pretrained ResNet18 checkpoints can be loaded and used for inference. Include at least one smoke test per model.

---

## 8. Hyperparameter tuning requirement — important

The project specification requires **both Grid Search and Random Search** in Milestone 1 and at least one comparison between them (e.g. best score, number of trials, time).

The existing WD-0001, WD-001, and WD-01 runs are exploratory weight-decay experiments. They should not automatically be described as a completed Grid Search, and no Random Search has been completed in the work recorded here.

Recommended controlled scope:
- Tune only Learning Rate and Weight Decay.
- Use the same model, split, data transforms, batch size, epoch budget, and evaluation metric for both methods.
- Use Macro F1 as the primary selection metric; also report validation loss and accuracy.
- Record number of trials and wall-clock time for each method.
- Agree on a realistic trial budget and GPU availability before launching more training.

Do not add dropout or change the architecture just for tuning. The project says only a subset of hyperparameters needs to be tuned.

---

## 9. Suggested immediate checklist

1. Omar checks `git status`, confirms the notebook is saved, and stages only the intended notebook and handoff document.
2. Omar commits and pushes those handoff files to `transfer-learning`; do not merge yet.
3. Share the repository branch, Colab notebook, and Google Drive artifacts with the next developer; confirm Editor access to the Drive folder if they need to add files.
4. The next developer pulls `origin/transfer-learning` and continues the outstanding transfer-learning work on that same branch.
5. Evaluate WD-01 from the saved best checkpoint; do not retrain it just to obtain evaluation metrics.
6. Complete a documented Grid Search and Random Search comparison using a fair, agreed trial budget and consistent validation setup.
7. When transfer-learning is complete, open a PR from `transfer-learning` to `main`, review it, and merge it.
8. After that merge, create `workstream-d-integration` from the updated `main`.
9. Agree on the encoder interface with Hossam and implement Workstream D on the integration branch: Grad-CAM, CLI, experiment tracking coordination, inference smoke tests, and reproducibility.
10. Open a separate PR from `workstream-d-integration` to `main`; update the README and report with exact commands, results, limitations, and contributions.

---

## 10. Important constraints and cautions

- Preserve the original EDA split: 4,795 train / 1,198 validation.
- Do not commit datasets or large `.pth` checkpoints to Git.
- Keep notebooks for experiments; final implementation must be Python modules callable through the CLI.
- Do not silently change preprocessing or transforms between comparison experiments.
- Do not claim the hyperparameter-tuning requirement is complete until both search methods and their comparison are documented.
- Do not claim a checkpoint can resume training unless the code supports it.
- Keep the validation results labelled as validation results, not test results.
- The current best transfer-learning result is Fine-Tuning: 67.70% accuracy and 67.48% Macro F1.
- The next developer should understand and review the code before making changes; this handoff is a guide, not a substitute for inspecting the repository and notebook.

