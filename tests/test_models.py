"""Tests for the shared classifier interface (src/models/registry.py).

pretrained=False everywhere so the tests never download ImageNet weights.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from src.models import (
    available_models,
    build_classifier,
    get_gradcam_target_layer,
    load_classifier,
)


class ModelRegistryTests(unittest.TestCase):
    def test_resnet18_is_registered(self):
        self.assertIn("resnet18", available_models())

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            build_classifier("not_a_model", pretrained=False)

    def test_output_follows_contract(self):
        model = build_classifier("resnet18", pretrained=False).eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(logits.shape, (2, 200))

    def test_gradcam_layer_is_last_resnet_block(self):
        model = build_classifier("resnet18", pretrained=False)
        self.assertIs(get_gradcam_target_layer("resnet18", model), model.layer4[-1])


class LoadClassifierTests(unittest.TestCase):
    def _save(self, obj, directory):
        path = Path(directory) / "best_model.pth"
        torch.save(obj, path)
        return path

    def test_loads_training_checkpoint_format(self):
        """Same dict layout that src/transfer_learning/train.py saves."""
        source = build_classifier("resnet18", pretrained=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._save(
                {
                    "epoch": 10,
                    "model_state_dict": source.state_dict(),
                    "optimizer_state_dict": {},
                    "mode": "fine_tuning",
                    "learning_rate": 1e-4,
                    "weight_decay": 0.0,
                    "batch_size": 32,
                },
                tmp,
            )
            model, meta = load_classifier(path)

        self.assertFalse(model.training)
        self.assertEqual(meta["epoch"], 10)
        self.assertEqual(meta["mode"], "fine_tuning")
        self.assertNotIn("optimizer_state_dict", meta)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x), source.eval()(x)))

    def test_loads_bare_state_dict(self):
        source = build_classifier("resnet18", pretrained=False)
        with tempfile.TemporaryDirectory() as tmp:
            model, meta = load_classifier(self._save(source.state_dict(), tmp))
        self.assertEqual(meta["model_name"], "resnet18")

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_classifier("does/not/exist.pth")

    def test_wrong_number_of_classes_fails_loudly(self):
        source = build_classifier("resnet18", num_classes=10, pretrained=False)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                load_classifier(self._save(source.state_dict(), tmp))


if __name__ == "__main__":
    unittest.main()
