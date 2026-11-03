"""Smoke tests for single-image prediction (no downloads, random weights)."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.inference import get_eval_transform, load_class_names, predict_image
from src.models import build_classifier


class PredictTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_classifier("resnet18", pretrained=False).eval()
        self.names = [f"species_{i}" for i in range(200)]

    def test_eval_transform_output_shape(self):
        tensor = get_eval_transform("resnet18")(Image.new("RGB", (500, 333)))
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_predict_image_returns_sorted_top_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bird.jpg"
            Image.new("RGB", (300, 200), (120, 80, 40)).save(path)
            results = predict_image(self.model, "resnet18", path, self.names, top_k=5)

        self.assertEqual(len(results), 5)
        probs = [r["probability"] for r in results]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertEqual(results[0]["species"], self.names[results[0]["class_index"]])

    def test_class_names_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "names.json"
            path.write_text(json.dumps(self.names))
            self.assertEqual(load_class_names(path), self.names)

    def test_class_names_file_with_wrong_length_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "names.json"
            path.write_text(json.dumps(["a", "b"]))
            with self.assertRaises(ValueError):
                load_class_names(path)


if __name__ == "__main__":
    unittest.main()
