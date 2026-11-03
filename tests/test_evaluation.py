"""Tests for src/evaluation (hand-checkable numbers, no downloads)."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation import compute_metrics, evaluate_model


def one_hot_logits(predictions, num_classes):
    logits = torch.full((len(predictions), num_classes), -5.0)
    logits[range(len(predictions)), predictions] = 5.0
    return logits


class ComputeMetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        labels = torch.tensor([0, 1, 2, 3])
        m = compute_metrics(one_hot_logits(labels, 4), labels, num_classes=4)
        for key in ("accuracy", "top5_accuracy", "macro_f1", "weighted_f1"):
            self.assertAlmostEqual(m[key], 1.0, msg=key)

    def test_known_values(self):
        # 4 classes, true = [0,0,1,2], pred = [0,1,1,0]
        labels = torch.tensor([0, 0, 1, 2])
        m = compute_metrics(one_hot_logits([0, 1, 1, 0], 4), labels, num_classes=4)
        self.assertAlmostEqual(m["accuracy"], 0.5)
        # per class F1: c0 = 0.5, c1 = 2/3, c2 = 0, c3 = 0 (never seen, never predicted)
        self.assertAlmostEqual(m["macro_f1"], (0.5 + 2 / 3) / 4)
        self.assertEqual(m["num_samples"], 4)

    def test_top5_counts_true_class_in_top_five(self):
        logits = torch.arange(10, dtype=torch.float).repeat(2, 1)  # top5 = 5..9
        labels = torch.tensor([7, 2])
        m = compute_metrics(logits, labels, num_classes=10)
        self.assertAlmostEqual(m["top5_accuracy"], 0.5)
        self.assertAlmostEqual(m["accuracy"], 0.0)


class ConstantModel(torch.nn.Module):
    """Always predicts class 1, so results are predictable."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 3)
        logits[:, 1] = 1.0
        return logits + self.dummy


class EvaluateModelTests(unittest.TestCase):
    def test_writes_three_result_files(self):
        samples = [{"image": torch.zeros(3, 8, 8), "label": torch.tensor(i % 3)}
                   for i in range(6)]
        loader = DataLoader(samples, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            m = evaluate_model(ConstantModel(), loader, ["a", "b", "c"], tmp,
                               extra_info={"mode": "fine_tuning"})
            out = Path(tmp)
            saved = json.loads((out / "evaluation_metrics.json").read_text())
            matrix = np.loadtxt(out / "confusion_matrix.csv", delimiter=",")
            rows = list(csv.DictReader((out / "validation_predictions.csv").open()))

        self.assertAlmostEqual(m["accuracy"], 1 / 3)
        self.assertEqual(saved["mode"], "fine_tuning")
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(matrix[:, 1].sum(), 6)       # every image predicted as class 1
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["pred_species"], "b")


if __name__ == "__main__":
    unittest.main()
