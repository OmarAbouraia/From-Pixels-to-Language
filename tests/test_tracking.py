"""Tests for MLflow tracking using fake run folders (no training)."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.tracking import collect_run_info, export_summary, find_run_folders, log_runs


def make_run(folder: Path, evaluated=True, trial=False):
    folder.mkdir(parents=True)
    with (folder / "training_history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss",
                         "val_accuracy"])
        for epoch in (1, 2, 3):
            writer.writerow([epoch, 2.0 / epoch, 0.3 * epoch, 2.5 / epoch, 0.2 * epoch])
    if evaluated:
        (folder / "evaluation_metrics.json").write_text(json.dumps({
            "epoch": 3, "mode": "fine_tuning", "learning_rate": 0.0001,
            "weight_decay": 0.01, "batch_size": 32, "accuracy": 0.6336,
            "macro_f1": 0.6316, "val_loss": 1.5584,
            "checkpoint_path": "/drive/WD-01/best_model.pth"}))
    if trial:
        (folder / "trial_result.json").write_text(json.dumps({
            "method": "grid", "trial": 1, "learning_rate": 0.0003,
            "weight_decay": 0.0, "train_seconds": 312.5, "macro_f1": 0.65,
            "accuracy": 0.66, "val_loss": 1.3}))
    (folder / "best_model.pth").write_bytes(b"not uploaded")


class CollectRunInfoTests(unittest.TestCase):
    def test_reads_params_metrics_and_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "WD-01"
            make_run(run)
            info = collect_run_info(run)
        self.assertEqual(info["params"]["weight_decay"], 0.01)
        self.assertEqual(info["params"]["best_epoch"], 3)
        self.assertEqual(info["params"]["epochs_trained"], 3)
        self.assertAlmostEqual(info["metrics"]["macro_f1"], 0.6316)
        self.assertNotIn("checkpoint_path", info["metrics"])
        self.assertEqual(len(info["history"]), 3)
        self.assertTrue(info["evaluated"])

    def test_unevaluated_folder_infers_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "Feature-Extraction"
            make_run(run, evaluated=False)
            info = collect_run_info(run)
        self.assertEqual(info["params"]["mode"], "feature_extraction")
        self.assertFalse(info["evaluated"])

    def test_trial_values_are_tracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "grid" / "trial_01"
            make_run(run, evaluated=False, trial=True)
            info = collect_run_info(run)
        self.assertEqual(info["params"]["method"], "grid")
        self.assertEqual(info["metrics"]["train_seconds"], 312.5)


class LogRunsTests(unittest.TestCase):
    def test_logs_all_runs_once_and_exports(self):
        import mlflow

        with tempfile.TemporaryDirectory() as tmp:
            root, tracking = Path(tmp) / "runs", Path(tmp) / "mlflow"
            make_run(root / "Fine-Tuning")
            make_run(root / "Tuning" / "grid" / "trial_01", evaluated=False, trial=True)
            self.assertEqual(len(find_run_folders(root)), 2)

            self.assertEqual(len(log_runs(root, tracking)), 2)
            self.assertEqual(log_runs(root, tracking), [])          # rerun: skipped

            mlflow.set_tracking_uri(f"sqlite:///{tracking / 'mlflow.db'}")
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name("classification")
            runs = client.search_runs([exp.experiment_id])
            self.assertEqual(len(runs), 2)
            run = next(r for r in runs if r.info.run_name == "Fine-Tuning")
            self.assertEqual(run.data.params["weight_decay"], "0.01")
            self.assertEqual(len(client.get_metric_history(run.info.run_id,
                                                           "val_loss")), 4)  # 3 epochs + final
            self.assertTrue(run.data.tags["checkpoint_path"].endswith("best_model.pth"))
            artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
            self.assertIn("training_history.csv", artifacts)
            self.assertNotIn("best_model.pth", artifacts)            # too large

            rows = export_summary(tracking)
            self.assertEqual(len(rows), 2)
            self.assertTrue((tracking / "runs_summary.csv").exists())
            self.assertTrue((tracking / "tracked_runs_curves.png").exists())


class TrackingFolderInsideRunsTests(unittest.TestCase):
    def test_tracking_folder_is_not_scanned(self):
        """Regression: tracking dir inside the scanned folder must not create
        extra runs from MLflow's own artifact copies."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            tracking = root / "mlflow"
            make_run(root / "Fine-Tuning")
            self.assertEqual(len(log_runs(root, tracking)), 1)
            # artifacts now contain a training_history.csv copy
            self.assertTrue(list(tracking.rglob("training_history.csv")))
            self.assertEqual(log_runs(root, tracking), [])
            self.assertEqual(len(find_run_folders(root, exclude=tracking)), 1)


if __name__ == "__main__":
    unittest.main()
