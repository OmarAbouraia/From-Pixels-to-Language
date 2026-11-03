"""CLI tests: argument parsing, config handling and an end-to-end predict run."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from src import cli
from src.models import build_classifier

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_all_configs_load(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            config = cli.load_config(path)
            self.assertEqual(config["task"], "classification", path.name)

    def test_train_calls_existing_pipeline_with_config_and_overrides(self):
        config = ROOT / "configs" / "classification_fine_tuning.yaml"
        with patch("src.transfer_learning.train.run_experiment") as run:
            code = cli.main(["train", "--config", str(config), "--epochs", "1"])
        self.assertEqual(code, 0)
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["mode"], "fine_tuning")
        self.assertEqual(kwargs["epochs"], 1)            # override wins
        self.assertEqual(kwargs["learning_rate"], 0.0001)

    def test_train_scratch_reports_not_connected(self):
        config = ROOT / "configs" / "classification_scratch.yaml"
        self.assertEqual(cli.main(["train", "--config", str(config)]), 2)

    def test_predict_end_to_end(self):
        torch.manual_seed(0)
        model = build_classifier("resnet18", pretrained=False)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "best_model.pth"
            torch.save({"epoch": 3, "model_state_dict": model.state_dict(),
                        "mode": "fine_tuning"}, ckpt)
            image = Path(tmp) / "bird.png"
            Image.new("RGB", (256, 256), (10, 200, 30)).save(image)
            names = Path(tmp) / "names.json"
            names.write_text(str([f"s{i}" for i in range(200)]).replace("'", '"'))
            out = Path(tmp) / "pred.json"

            code = cli.main(["predict", "--checkpoint", str(ckpt), "--image",
                             str(image), "--class-names", str(names),
                             "--device", "cpu", "--top-k", "3", "--json", str(out)])
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())


class EvaluateCommandTests(unittest.TestCase):
    def test_evaluate_end_to_end(self):
        from torch.utils.data import DataLoader

        torch.manual_seed(0)
        model = build_classifier("resnet18", pretrained=False)
        samples = [{"image": torch.randn(3, 64, 64), "label": torch.tensor(i)}
                   for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "best_model.pth"
            torch.save({"epoch": 7, "model_state_dict": model.state_dict(),
                        "mode": "fine_tuning", "weight_decay": 0.01}, ckpt)
            out = Path(tmp) / "eval"
            with patch("src.cli.get_val_loader",
                       return_value=DataLoader(samples, batch_size=2)), \
                    patch("src.inference.load_class_names",
                          return_value=[f"s{i}" for i in range(200)]):
                code = cli.main(["evaluate", "--checkpoint", str(ckpt),
                                 "--device", "cpu", "--output-dir", str(out)])
            self.assertEqual(code, 0)
            saved = (out / "evaluation_metrics.json").read_text()
            self.assertIn('"weight_decay": 0.01', saved)   # checkpoint info kept
            self.assertTrue((out / "confusion_matrix.csv").exists())


if __name__ == "__main__":
    unittest.main()
