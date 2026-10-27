# Project Architecture — From Pixels to Language

## 1. Purpose and current scope

The project uses the CUB-200-2011 bird dataset and develops three related capabilities across the project: bird-species classification, image colorization, and image captioning.

**Current milestone focus:** understand and audit the dataset, build a classification baseline, compare a CNN trained from scratch with transfer learning, tune hyperparameters, track experiments, and visualize model attention with Grad-CAM.

This document describes the proposed architecture. Decisions marked **Proposed** should be reviewed and agreed on by the team before they are treated as final.

## 2. High-level flow

```text
CUB-200-2011 files + annotations
              |
              v
      Dataset / Data Audit
              |
              v
      Split + Preprocessing
              |
              v
         DataLoaders
              |
              v
     Classification Encoder
       /               \
 CNN from scratch   Transfer learning
       \               /
              v
       Training / Tuning
              |
              v
      Validation + Metrics
              |
              v
     Best checkpoint + report
              |
              v
       Grad-CAM analysis
```

## 3. Suggested repository responsibilities

- `src/data/`: reading annotations, mapping image paths and labels, transforms, dataset and DataLoader creation.
- `src/models/`: model construction and a consistent model-output interface.
- `src/training/`: training loop, optimizer, scheduler, checkpoint saving and experiment configuration.
- `src/evaluation/`: metrics, predictions, per-class results and confusion matrix.
- `src/visualization/`: training curves, sample predictions and Grad-CAM.
- `configs/`: shared, version-controlled experiment settings.
- `notebooks/`: exploratory analysis and one-off experiments; reusable pipeline logic should move into `src/`.
- `tests/`: small tests for label mapping, data shapes, model output and metric calculation.
- `docs/`: team decisions, interfaces and development workflow.
- `outputs/`: local/generated outputs only; large checkpoints and generated artifacts should not be committed to Git.

## 4. Dataset facts and boundaries

CUB-200-2011 contains approximately 11,788 bird images across 200 bird-species classes, with the official dataset providing train/test indicators and annotations. Images do **not** all have to share the same raw width and height.

The data pipeline should use the official train/test designation. Create a validation subset from the official training portion; do not use the official test set to choose hyperparameters, select checkpoints, or make repeated development decisions.

Before implementation, the team should inspect the actual downloaded files and verify:
- annotation files and image paths are complete;
- label IDs are mapped consistently to class names;
- the official train/test indicator is parsed correctly;
- whether bounding-box/crop annotations are used or the full image is classified;
- class counts and any split imbalance.

## 5. Milestone 1 deliverables

1. Dataset audit / EDA report.
2. Reproducible train-validation-test pipeline.
3. CNN-from-scratch baseline.
4. Transfer-learning baseline.
5. Hyperparameter-search results and experiment log.
6. Validation/test metrics, with test evaluation performed only after model selection.
7. Grad-CAM examples and a short interpretation.

## 6. Future milestones

Colorization and captioning should be added as separate modules when their requirements and evaluation methods are clear. They should reuse shared dataset utilities where appropriate, but they should not force the current classification API to return caption text or colorized images.
