"""Grad-CAM (Selvaraju et al., 2017) for any registered classifier.

Idea in four steps:
  1. Forward the image and keep the feature maps A of the target conv layer
     (shape [1, K, h, w]; for ResNet18 layer4 that is [1, 512, 7, 7]).
  2. Back-propagate the score (logit) of the class we want to explain and
     keep dScore/dA.
  3. Average those gradients over h and w -> one weight per channel (alpha_k).
     A large alpha_k means channel k pushes the score up.
  4. CAM = ReLU(sum_k alpha_k * A_k), upsampled to the input size and scaled
     to [0, 1]. ReLU keeps only regions that support the class.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transfer_learning.transforms import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Usage:

        with GradCAM(model, target_layer) as cam:
            heatmap, predicted_class = cam(image_tensor)            # predicted class
            heatmap, _ = cam(image_tensor, class_index=true_label)  # any class
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        # Hook on the output tensor itself to receive its gradient on backward.
        # Skipped under torch.no_grad() so normal inference still works.
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient.detach()

    def __call__(self, image_tensor: torch.Tensor, class_index: int | None = None):
        """Return (heatmap [H, W] numpy array in [0, 1], explained class index)."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        device = next(self.model.parameters()).device
        # The input must require grad: in feature-extraction checkpoints the
        # backbone is frozen, so otherwise layer4 would receive no gradient.
        image_tensor = image_tensor.to(device).clone().requires_grad_(True)

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)

        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[0, class_index].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)        # [1, K, 1, 1]
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:],
                            mode="bilinear", align_corners=False)[0, 0]

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy(), class_index

    def remove(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def tensor_to_display_image(image_tensor: torch.Tensor,
                            normalization: str = "imagenet") -> np.ndarray:
    """Undo normalisation so the image can be shown: [3,H,W] -> [H,W,3] in [0,1]."""
    image = image_tensor.detach().cpu().squeeze(0).clone()
    if normalization == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image = image * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    """Blend a [0,1] heatmap (jet colour map) on top of an RGB image."""
    import matplotlib

    colored = matplotlib.colormaps["jet"](heatmap)[..., :3]
    return np.clip((1 - alpha) * image + alpha * colored, 0, 1)


def save_gradcam_figure(image: np.ndarray, heatmap: np.ndarray, path,
                        title: str = ""):
    """Save a 3-panel figure: original | heatmap | overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    panels = [(image, "Input"), (heatmap, "Grad-CAM"),
              (overlay_heatmap(image, heatmap), "Overlay")]
    for axis, (content, name) in zip(axes, panels):
        axis.imshow(content, cmap="jet" if content.ndim == 2 else None)
        axis.set_title(name)
        axis.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
