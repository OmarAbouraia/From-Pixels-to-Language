# Interface Contracts — From Pixels to Language

## Status

This is a **proposal for team review**, not a claim that every setting is already decided. The aim is to agree on stable boundaries between teammates' modules so work can happen in parallel.

## 1. Core principle

Keep the public interfaces small and predictable. Internal implementation can differ, but each module must respect the agreed inputs, outputs, label mapping and configuration.

## 2. Dataset and batch contract

### Raw dataset
- Source: CUB-200-2011 files and annotation files.
- Each example represents one image and one bird-species class for the classification task.
- Keep a stable mapping from dataset class ID to class name. The dataset has 200 classes.
- Preserve the official test designation; derive validation data only from the official training portion.
- Raw images can have different widths and heights. Do not assume every file is already 224×224.

### Proposed model-ready batch
For the first classification baseline, propose a fixed input size of **224 × 224** and RGB images:

- `images`: floating-point tensor with shape `[B, 3, 224, 224]`.
- `labels`: integer tensor with shape `[B]`, values in `[0, 199]` if using zero-based labels.
- `B`: current batch size; it can vary, especially for the final batch.
- Channel order: RGB.
- Image tensor layout: NCHW (`batch, channels, height, width`), which is the usual PyTorch convention.
- Labels use `torch.int64` / `long` for standard cross-entropy classification.

**Important distinction:** the `3` is the number of color channels (R, G, B); `B` is the number of images in a batch. They are different dimensions.

Example: with `B=32`, a batch has image shape `[32, 3, 224, 224]` and label shape `[32]`.

### Preprocessing decisions to agree on
1. **Input size:** 224×224 is a practical starting proposal, not an immutable fact. Confirm it against the selected architectures and available compute.
2. **Training transforms:** likely resize/crop plus modest augmentation (e.g. horizontal flip); choose and document the exact transforms.
3. **Validation/test transforms:** deterministic; no random augmentation.
4. **Normalization:** define mean/std in one shared config. For pretrained models, use the preprocessing expected by the chosen pretrained weights. If scratch and pretrained models need different normalization, make it model/config-specific rather than silently changing the shared data contract.
5. **Full image vs annotated bird crop:** decide whether classification uses the full image or CUB bounding-box crops. Apply the same policy consistently to all compared models, or explicitly document a controlled comparison.

The dataset module should return model-ready tensors. It should not decide the model architecture or compute training metrics.

## 3. Model contract

A classifier accepts `images` shaped `[B, 3, H, W]` and returns **raw logits**, not softmax probabilities:

- Output: `logits` shaped `[B, 200]`.
- One row per input image; one column per class.
- Class-column order must match the shared label mapping.
- Training can pass logits directly to `CrossEntropyLoss`.
- Apply softmax only when probabilities are specifically needed for reporting or visualization.

This lets the training loop work with either the scratch CNN or transfer-learning model without special-case logic.

## 4. Training contract

The training component receives:
- model;
- training and validation DataLoaders;
- loss function;
- optimizer and optional scheduler;
- configuration (epochs, device, checkpoint path, seed, etc.).

It should:
- train on the training loader;
- evaluate on validation data at the agreed interval;
- record loss and agreed metrics by epoch;
- save checkpoints according to a clearly stated selection rule, normally best validation metric;
- return or save a structured history that can be plotted later.

The training loop should not silently evaluate the test set every epoch.

## 5. Evaluation contract

For this single-label, 200-class classification task, report:
- **Top-1 accuracy** as a primary metric;
- **macro precision, macro recall and macro F1** so each class contributes equally to the aggregate;
- optionally weighted F1 and per-class precision/recall/F1 for deeper analysis;
- confusion matrix when useful and feasible to inspect;
- validation loss and test loss if implemented consistently.

Do not assume precision, recall and F1 are binary metrics here. Use a clearly stated multiclass averaging method (`macro` / `weighted`) and a consistent label order. Define how zero-division cases are handled.

**Evaluation protocol:**
- Use validation metrics for model selection and hyperparameter tuning.
- Evaluate the selected final model on the test set at the end.
- Do not choose a checkpoint or tune settings based on test results.

## 6. Configuration contract

Put experiment settings in config files rather than scattering magic numbers through modules. At minimum, record:
- dataset root and split strategy;
- image size and transforms;
- batch size;
- model name and pretrained-weights choice;
- optimizer, learning rate, weight decay and scheduler;
- number of epochs and early-stopping/checkpoint-selection rule;
- random seed;
- device;
- output/checkpoint directory;
- metrics to log.

Suggested first defaults for discussion only: `image_size=224`, `batch_size=32`, `num_workers` chosen for the environment, and a fixed seed. These are starting points to test, not guaranteed optimal values.

## 7. Experiment output contract

Every run should make it possible to identify:
- experiment/run name;
- config used;
- model and pretrained-weight setting;
- random seed;
- training/validation history;
- final validation metrics;
- checkpoint path;
- code revision/commit where practical.

Save large checkpoints and bulky outputs outside Git (e.g. shared Drive or an experiment storage location). Commit lightweight configs and reports, not large binary files.

## 8. CLI / entry-point contract

Prefer a small command-line entry point for reproducible runs. Exact command syntax is to be agreed on, but it should allow the user to specify a config and output/run directory without editing source code.

Illustrative target interface (not implemented yet):

```bash
python -m src.training.train --config configs/classification_baseline.yaml
python -m src.evaluation.evaluate --checkpoint <checkpoint_path> --config <config_path>
```

## 9. Decisions that require team agreement

Use a decision table in the meeting and update this file after agreement.

| Decision | Proposed starting point | Why it matters |
|---|---|---|
| Classification target | 200 species classes, zero-based IDs | Keeps labels and logits aligned |
| Raw image size | Variable; normalize in transforms | Raw files are not guaranteed to match |
| Model input size | 224×224 | Common starting point for CNN/transfer learning |
| Channel order/layout | RGB, NCHW | Prevents shape/channel bugs |
| Full image or bird crop | Decide before comparing models | Changes task difficulty and data pipeline |
| Normalization | Shared config; pretrained weights may require specific values | Avoids incompatible preprocessing |
| Validation split | Derived from official train split | Prevents test leakage |
| Batch size | Configurable; start around 32 if memory permits | GPU memory and model differ |
| Model output | Raw logits `[B, 200]` | One common training interface |
| Main metrics | Top-1 accuracy + macro P/R/F1 | Useful overall and class-balanced view |
| Checkpoint selection | Best validation metric, declared in advance | Reproducible model selection |
| Randomness | Fixed and recorded seed | Supports repeatability |
| Experiment tracking | Config + logs + metrics + checkpoint path | Enables fair comparison |

## 10. Future task interfaces

For colorization and captioning, define separate contracts when those milestones begin:
- **Colorization:** input representation, color-space choice, target/output shape, value range, loss and image-quality metrics.
- **Captioning:** image input, tokenization/vocabulary, caption target format, padding/masking, generated-token format and text metrics.

Do not decide those details only by copying the classification interface; the tasks have different data and evaluation needs.
