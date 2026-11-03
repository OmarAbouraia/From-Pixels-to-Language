"""Evaluate a trained classifier on the validation split.

Produces the same metrics already reported for Feature Extraction and
Fine-Tuning, so new results (WD-01, tuning trials) are directly comparable:

  accuracy        - fraction of images whose top prediction is correct
  top5_accuracy   - fraction whose true class is among the 5 highest logits
  macro_*         - precision/recall/F1 computed per class, then averaged
                    equally over the 200 classes (every species counts the same)
  weighted_*      - per-class scores averaged by number of images per class
  val_loss        - mean cross-entropy, same as during training

Classes that are never predicted get precision 0 (zero_division=0), so the
macro scores are not inflated.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@torch.no_grad()
def collect_outputs(model: nn.Module, dataloader, device="cpu"):
    """Return (logits [N, C], labels [N]) for the whole loader, in order."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in dataloader:
        all_logits.append(model(batch["image"].to(device)).cpu())
        all_labels.append(batch["label"].cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor,
                    num_classes: int = 200) -> dict:
    """All classification metrics from raw logits and integer labels."""
    predictions = logits.argmax(dim=1)
    y_true, y_pred = labels.numpy(), predictions.numpy()
    class_ids = list(range(num_classes))

    top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
    metrics = {
        "num_samples": int(labels.numel()),
        "val_loss": float(nn.functional.cross_entropy(logits, labels)),
        "accuracy": float((predictions == labels).float().mean()),
        "top5_accuracy": float((top5 == labels.unsqueeze(1)).any(dim=1).float().mean()),
    }
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=class_ids, average=average, zero_division=0
        )
        metrics[f"{average}_precision"] = float(precision)
        metrics[f"{average}_recall"] = float(recall)
        metrics[f"{average}_f1"] = float(f1)
    return metrics


def evaluate_model(model, dataloader, class_names, output_dir=None,
                   device="cpu", extra_info=None) -> dict:
    """Evaluate and (optionally) save the three standard result files:

        evaluation_metrics.json   - metrics above + checkpoint info
        confusion_matrix.csv      - 200 x 200, rows = true, columns = predicted
        validation_predictions.csv - one row per validation image
    """
    num_classes = len(class_names)
    logits, labels = collect_outputs(model, dataloader, device)
    metrics = compute_metrics(logits, labels, num_classes)
    if extra_info:
        metrics = {**extra_info, **metrics}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "evaluation_metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8")

        matrix = confusion_matrix(labels.numpy(), logits.argmax(1).numpy(),
                                  labels=list(range(num_classes)))
        np.savetxt(output_dir / "confusion_matrix.csv", matrix,
                   fmt="%d", delimiter=",")

        probabilities = torch.softmax(logits, dim=1)
        dataset_indices = getattr(dataloader.dataset, "indices",
                                  list(range(len(labels))))
        with (output_dir / "validation_predictions.csv").open(
                "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["val_position", "dataset_index", "true_index",
                             "true_species", "pred_index", "pred_species",
                             "confidence", "correct"])
            for position in range(len(labels)):
                true_idx = int(labels[position])
                pred_idx = int(logits[position].argmax())
                writer.writerow([
                    position, dataset_indices[position], true_idx,
                    class_names[true_idx], pred_idx, class_names[pred_idx],
                    f"{float(probabilities[position, pred_idx]):.6f}",
                    int(true_idx == pred_idx),
                ])
        print(f"Saved evaluation files to {output_dir}")

    return metrics


def print_metrics(metrics: dict) -> None:
    print("\nEvaluation results (validation split)")
    print("-" * 40)
    for key in ("val_loss", "accuracy", "top5_accuracy", "macro_precision",
                "macro_recall", "macro_f1", "weighted_precision",
                "weighted_recall", "weighted_f1"):
        value = metrics[key]
        text = f"{value:.4f}" if key == "val_loss" else f"{value * 100:.2f}%"
        print(f"{key:<20} {text}")
