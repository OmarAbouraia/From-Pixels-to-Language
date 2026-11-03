"""Generate Grad-CAM figures for correct AND incorrect validation predictions.

Uses the original EDA validation split (1,198 images) via Omar's loader, so
results are comparable with the reported metrics. Nothing is retrained.
"""

import csv
import re
from pathlib import Path

import torch

from src.models import get_gradcam_target_layer, get_model_spec
from src.visualization.gradcam import (
    GradCAM,
    save_gradcam_figure,
    tensor_to_display_image,
)


def _short(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:40]


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """Run the model over a loader; return (images, labels, predictions)."""
    model.eval()
    images, labels, predictions = [], [], []
    for batch in dataloader:
        logits = model(batch["image"].to(device))
        images.append(batch["image"])
        labels.append(batch["label"])
        predictions.append(logits.argmax(dim=1).cpu())
    return torch.cat(images), torch.cat(labels), torch.cat(predictions)


def generate_gradcam_examples(model, model_name, dataloader, class_names,
                              output_dir, num_correct=8, num_incorrect=8,
                              device="cpu"):
    """Save Grad-CAM figures plus a summary CSV; return the CSV rows."""
    output_dir = Path(output_dir)
    (output_dir / "correct").mkdir(parents=True, exist_ok=True)
    (output_dir / "incorrect").mkdir(parents=True, exist_ok=True)

    images, labels, predictions = collect_predictions(model, dataloader, device)
    correct = (labels == predictions).nonzero().flatten()[:num_correct]
    incorrect = (labels != predictions).nonzero().flatten()[:num_incorrect]
    print(f"Validation accuracy on this loader: "
          f"{(labels == predictions).float().mean():.4f}")

    normalization = get_model_spec(model_name).normalization
    target_layer = get_gradcam_target_layer(model_name, model)
    rows = []

    with torch.enable_grad(), GradCAM(model, target_layer) as cam:
        for group, indices in (("correct", correct), ("incorrect", incorrect)):
            for position in indices.tolist():
                true_idx = int(labels[position])
                pred_idx = int(predictions[position])
                image = images[position]
                display = tensor_to_display_image(image, normalization)

                heatmap, _ = cam(image, class_index=pred_idx)
                stem = f"val{position:04d}_true-{_short(class_names[true_idx])}"
                title = (f"[{group}] true: {class_names[true_idx]} | "
                         f"pred: {class_names[pred_idx]}")
                path = output_dir / group / f"{stem}.png"
                save_gradcam_figure(display, heatmap, path, title=title)

                # For mistakes, also show what the model "sees" for the TRUE
                # class: useful to discuss why it went wrong.
                true_path = ""
                if group == "incorrect":
                    true_heatmap, _ = cam(image, class_index=true_idx)
                    true_path = output_dir / group / f"{stem}_cam-for-true.png"
                    save_gradcam_figure(
                        display, true_heatmap, true_path,
                        title=f"Grad-CAM for TRUE class: {class_names[true_idx]}",
                    )

                rows.append({
                    "group": group,
                    "val_position": position,
                    "true_index": true_idx,
                    "true_species": class_names[true_idx],
                    "pred_index": pred_idx,
                    "pred_species": class_names[pred_idx],
                    "figure": str(path),
                    "figure_true_class": str(true_path),
                })

    with (output_dir / "gradcam_summary.csv").open("w", newline="",
                                                   encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys())
                                if rows else ["group"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} Grad-CAM examples to {output_dir}")
    return rows
