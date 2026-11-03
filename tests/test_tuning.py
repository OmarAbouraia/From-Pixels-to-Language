"""Tests for Grid/Random Search. Training is replaced by a fake that saves a
small checkpoint, so these run in seconds on a CPU."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from src.models import build_classifier
from src.tuning import grid_trials, random_trials, run_tuning, summarize


CONFIG = {
    "model": "resnet18", "mode": "fine_tuning", "epochs": 1, "batch_size": 2,
    "seed": 42, "num_workers": 0, "selection_metric": "macro_f1",
    "keep_best_checkpoints": False,
    "grid": {"learning_rate": [1e-4, 1e-3], "weight_decay": [0.0]},
    "random": {"seed": 0,
               "learning_rate": {"low": 1e-5, "high": 1e-3},
               "weight_decay": {"low": 1e-6, "high": 1e-2}},
}


def fake_run_experiment(**kwargs):
    """Stands in for training: writes the files real training writes."""
    out = Path(kwargs["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model = build_classifier("resnet18", pretrained=False)
    checkpoint = {"epoch": 1, "model_state_dict": model.state_dict(),
                  "mode": kwargs["mode"]}
    torch.save(checkpoint, out / "best_model.pth")
    torch.save(checkpoint, out / "last_checkpoint.pth")
    return []


def fake_val_loader(*args, **kwargs):
    samples = [{"image": torch.randn(3, 64, 64), "label": torch.tensor(i)}
               for i in range(4)]
    return DataLoader(samples, batch_size=2)


class SearchSpaceTests(unittest.TestCase):
    def test_grid_has_every_combination(self):
        trials = grid_trials({"learning_rate": [1, 2, 3], "weight_decay": [0, 5]})
        self.assertEqual(len(trials), 6)
        self.assertEqual(len({tuple(t.values()) for t in trials}), 6)

    def test_random_is_reproducible_and_in_range(self):
        a = random_trials(CONFIG["random"], 20, seed=0)
        self.assertEqual(a, random_trials(CONFIG["random"], 20, seed=0))
        self.assertNotEqual(a, random_trials(CONFIG["random"], 20, seed=1))
        for t in a:
            self.assertTrue(1e-5 <= t["learning_rate"] <= 1e-3)
            self.assertTrue(1e-6 <= t["weight_decay"] <= 1e-2)

    def test_summary_finds_best_and_budget_to_reach_it(self):
        results = [{"macro_f1": f, "accuracy": f, "val_loss": 1.0,
                    "learning_rate": i, "weight_decay": 0.0, "train_seconds": 10.0}
                   for i, f in enumerate([0.5, 0.7, 0.6])]
        s = summarize("grid", results)
        self.assertEqual(s["best_macro_f1"], 0.7)
        self.assertEqual(s["trials_to_best"], 2)
        self.assertEqual(s["seconds_to_best"], 20.0)
        self.assertEqual(s["total_seconds"], 30.0)


class RunTuningTests(unittest.TestCase):
    def _run(self, tmp, methods="both"):
        config = {**CONFIG, "output_dir": tmp}
        with patch("src.transfer_learning.train.run_experiment",
                   side_effect=fake_run_experiment) as run, \
                patch("src.cli.get_val_loader", side_effect=fake_val_loader), \
                patch("src.inference.load_class_names",
                      return_value=[f"s{i}" for i in range(200)]):
            summaries = run_tuning(config, methods=methods, device="cpu")
        return summaries, run

    def test_both_methods_same_budget_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            summaries, run = self._run(tmp)
            out = Path(tmp)
            self.assertEqual(run.call_count, 4)          # 2 grid + 2 random
            self.assertEqual([s["num_trials"] for s in summaries], [2, 2])
            for name in ("grid_results.csv", "random_results.csv",
                         "comparison.csv", "comparison.json",
                         "tuning_comparison.png"):
                self.assertTrue((out / name).exists(), name)
            # large checkpoints removed as configured
            self.assertEqual(list(out.rglob("*.pth")), [])
            self.assertEqual(len(list(out.rglob("evaluation_metrics.json"))), 4)

    def test_rerun_skips_finished_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            _, run = self._run(tmp)
            self.assertEqual(run.call_count, 0)

    def test_separate_runs_are_still_compared(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp, methods="grid")
            summaries, _ = self._run(tmp, methods="random")
            self.assertEqual({s["method"] for s in summaries}, {"grid", "random"})
            data = json.loads((Path(tmp) / "comparison.json").read_text())
            self.assertEqual(len(data["summaries"]), 2)


if __name__ == "__main__":
    unittest.main()
