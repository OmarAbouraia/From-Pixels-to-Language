"""Tests for Grad-CAM (random weights, fake data, no downloads)."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models import build_classifier, get_gradcam_target_layer
from src.visualization import GradCAM, save_gradcam_figure, tensor_to_display_image
from src.visualization.gradcam_examples import generate_gradcam_examples


class GradCAMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_classifier("resnet18", pretrained=False).eval()
        self.layer = get_gradcam_target_layer("resnet18", self.model)

    def test_heatmap_shape_and_range(self):
        with GradCAM(self.model, self.layer) as cam:
            heatmap, cls = cam(torch.randn(3, 224, 224))
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertGreaterEqual(heatmap.min(), 0.0)
        self.assertLessEqual(heatmap.max(), 1.0)
        self.assertTrue(0 <= cls < 200)

    def test_default_class_is_prediction(self):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            expected = int(self.model(x).argmax())
        with GradCAM(self.model, self.layer) as cam:
            _, cls = cam(x)
        self.assertEqual(cls, expected)

    def test_captured_activation_shape(self):
        with GradCAM(self.model, self.layer) as cam:
            cam(torch.randn(1, 3, 224, 224), class_index=3)
            self.assertEqual(cam.activations.shape, (1, 512, 7, 7))
            self.assertEqual(cam.gradients.shape, (1, 512, 7, 7))

    def test_works_with_frozen_backbone(self):
        """Feature-extraction checkpoints freeze layer4; Grad-CAM must still work."""
        model = build_classifier("resnet18", mode="feature_extraction",
                                 pretrained=False).eval()
        layer = get_gradcam_target_layer("resnet18", model)
        with GradCAM(model, layer) as cam:
            heatmap, _ = cam(torch.randn(3, 224, 224))
        self.assertEqual(heatmap.shape, (224, 224))

    def test_hook_is_removed(self):
        with GradCAM(self.model, self.layer):
            pass
        self.assertEqual(len(self.layer._forward_hooks), 0)

    def test_save_figure(self):
        with GradCAM(self.model, self.layer) as cam:
            image = torch.randn(3, 224, 224)
            heatmap, _ = cam(image)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cam.png"
            save_gradcam_figure(tensor_to_display_image(image), heatmap, path, "t")
            self.assertGreater(path.stat().st_size, 0)


class GradCAMExamplesTests(unittest.TestCase):
    def test_generates_correct_and_incorrect_examples(self):
        torch.manual_seed(0)
        model = build_classifier("resnet18", pretrained=False).eval()
        x = torch.randn(6, 3, 224, 224)
        with torch.no_grad():
            preds = model(x).argmax(1)
        # First 3 labels match the prediction, last 3 do not.
        labels = torch.cat([preds[:3], (preds[3:] + 1) % 200])
        samples = [{"image": x[i], "label": labels[i]} for i in range(6)]
        loader = DataLoader(samples, batch_size=4)
        names = [f"species {i}" for i in range(200)]

        with tempfile.TemporaryDirectory() as tmp:
            rows = generate_gradcam_examples(model, "resnet18", loader, names, tmp,
                                             num_correct=2, num_incorrect=2)
            self.assertEqual([r["group"] for r in rows].count("correct"), 2)
            self.assertEqual([r["group"] for r in rows].count("incorrect"), 2)
            self.assertTrue((Path(tmp) / "gradcam_summary.csv").exists())
            self.assertEqual(len(list(Path(tmp).rglob("*.png"))), 6)  # 2 + 2*2


if __name__ == "__main__":
    unittest.main()
