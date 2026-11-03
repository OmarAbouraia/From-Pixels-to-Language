
"""
CNN V3 Training Script

Features:
- Residual CNN trained from scratch
- Strong data augmentation
- AdamW optimizer
- Label smoothing
- Weight decay
- Learning-rate warmup
- Cosine learning-rate decay
- Best model checkpoint
- Latest checkpoint for resuming
- Random-state recovery
- Training history
- CPU smoke test

Run from the repository root:
    python -m src.training.train_v3 --smoke-test --batch-size 2
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.data_loader_v3 import create_v3_data_loaders
from src.models.cnn_scratch_v3 import ScratchCNNV3


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state():
    """Save random states for checkpoint recovery."""

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state):
    """Restore random states after loading a checkpoint."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


# --------------------------------------------------
# 2. Training
# --------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (
            (logits.argmax(dim=1) == labels).sum().item()
        )
        total += batch_size

    return (
        total_loss / total,
        100.0 * correct / total
    )


# --------------------------------------------------
# 3. Validation
# --------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (
            (logits.argmax(dim=1) == labels).sum().item()
        )
        total += batch_size

    return (
        total_loss / total,
        100.0 * correct / total
    )


# --------------------------------------------------
# 4. Checkpoint Saving
# --------------------------------------------------

def save_checkpoint(path, checkpoint):
    """
    Save checkpoint using a temporary file first.

    This reduces the risk of corrupting an existing
    checkpoint if writing is interrupted.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


# --------------------------------------------------
# 5. Learning-Rate Schedule
# --------------------------------------------------

def get_lr_factor(epoch, total_epochs, warmup_epochs):
    """
    Linear warmup followed by cosine decay.

    Epoch is zero-indexed.
    """

    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs

    cosine_epochs = total_epochs - warmup_epochs

    progress = (
        epoch - warmup_epochs
    ) / max(1, cosine_epochs - 1)

    progress = min(max(progress, 0.0), 1.0)

    return 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


# --------------------------------------------------
# 6. Main Training Function
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=90
    )

    parser.add_argument(
        "--batch-size", type=int, default=32
    )

    parser.add_argument(
        "--lr", type=float, default=0.003
    )

    parser.add_argument(
        "--weight-decay", type=float, default=0.0001
    )

    parser.add_argument(
        "--dropout", type=float, default=0.3
    )

    parser.add_argument(
        "--label-smoothing", type=float, default=0.1
    )

    parser.add_argument(
        "--warmup-epochs", type=int, default=5
    )

    parser.add_argument(
        "--num-workers", type=int, default=0
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/cnn_v3"
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )

    parser.add_argument(
        "--max-run-epochs",
        type=int,
        default=None
    )

    args = parser.parse_args()

    if args.epochs <= args.warmup_epochs:
        raise ValueError(
            "Total epochs must exceed warmup epochs."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size must be positive."
        )

    if args.max_run_epochs is not None:
        if args.max_run_epochs <= 0:
            raise ValueError(
                "max-run-epochs must be positive."
            )

    # ----------------------------------------------
    # Set random seed
    # ----------------------------------------------

    set_seed(args.seed)

    # ----------------------------------------------
    # Select device
    # ----------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    # ----------------------------------------------
    # Load dataset
    # ----------------------------------------------

    train_loader, val_loader = create_v3_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation="strong"
    )

    print(
        "Training images:",
        len(train_loader.dataset),
        flush=True
    )

    print(
        "Validation images:",
        len(val_loader.dataset),
        flush=True
    )

    # ----------------------------------------------
    # Create CNN V3
    # ----------------------------------------------

    model = ScratchCNNV3(
        num_classes=200,
        dropout=args.dropout
    ).to(device)

    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters()),
        flush=True
    )

    # ----------------------------------------------
    # Loss function
    # ----------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )

    # ----------------------------------------------
    # AdamW optimizer
    # ----------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # ----------------------------------------------
    # Output paths
    # ----------------------------------------------

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_path = (
        output_dir / "latest_checkpoint.pth"
    )

    best_path = (
        output_dir / "best_model.pth"
    )

    history_path = (
        output_dir / "history.json"
    )

    config_path = (
        output_dir / "config.json"
    )

    # ----------------------------------------------
    # Training history
    # ----------------------------------------------

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    start_epoch = 0
    best_acc = -1.0
    best_epoch = 0

    # ----------------------------------------------
    # Resume training
    # ----------------------------------------------

    if args.resume:

        if not latest_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {latest_path}"
            )

        # Only load checkpoints you created and trust.
        checkpoint = torch.load(
            latest_path,
            map_location=device,
            weights_only=False
        )

        saved_config = checkpoint["config"]

        required_fields = [
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "dropout",
            "label_smoothing",
            "warmup_epochs",
            "seed",
        ]

        for field in required_fields:
            if saved_config[field] != getattr(args, field):
                raise ValueError(
                    f"Resume configuration mismatch: {field}"
                )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        history = checkpoint["history"]

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_acc = checkpoint["best_acc"]
        best_epoch = checkpoint["best_epoch"]

        if "rng_state" in checkpoint:
            restore_rng_state(
                checkpoint["rng_state"]
            )
        else:
            print(
                "Warning: Random state not found."
            )

        print(
            f"Resuming from epoch {start_epoch + 1}",
            flush=True
        )

    else:

        # Avoid silently overwriting an earlier run.
        if latest_path.exists():
            raise FileExistsError(
                f"Existing checkpoint found at {latest_path}. "
                "Use --resume or a new --output-dir."
            )

        config_path.write_text(
            json.dumps(vars(args), indent=2),
            encoding="utf-8"
        )

    # ----------------------------------------------
    # CPU/GPU smoke test
    # ----------------------------------------------

    if args.smoke_test:

        print(
            "Running one-batch forward/backward test...",
            flush=True
        )

        batch = next(iter(train_loader))

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "Non-finite loss detected."
            )

        loss.backward()
        optimizer.step()

        print(
            "Smoke test passed. Loss:",
            loss.item(),
            flush=True
        )

        return

    # ----------------------------------------------
    # Determine how many epochs to run this session
    # ----------------------------------------------

    end_epoch = args.epochs

    if args.max_run_epochs is not None:
        end_epoch = min(
            args.epochs,
            start_epoch + args.max_run_epochs
        )

    # ----------------------------------------------
    # Training loop
    # ----------------------------------------------

    for epoch in range(start_epoch, end_epoch):

        # Linear warmup + cosine decay
        factor = get_lr_factor(
            epoch,
            args.epochs,
            args.warmup_epochs
        )

        current_lr = args.lr * factor

        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Validate
        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        # Save metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Track best model
        improved = val_acc > best_acc

        if improved:
            best_acc = val_acc
            best_epoch = epoch + 1

        # Save checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "config": vars(args),
            "rng_state": get_rng_state(),
        }

        # Save best model
        if improved:
            save_checkpoint(
                best_path,
                checkpoint
            )

        # Save latest checkpoint every epoch
        save_checkpoint(
            latest_path,
            checkpoint
        )

        # Save history
        history_path.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8"
        )

        # Display epoch results
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"LR {current_lr:.6f} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.2f}% | "
            f"Best {best_acc:.2f}% "
            f"(epoch {best_epoch})",
            flush=True
        )

    # ----------------------------------------------
    # Final summary
    # ----------------------------------------------

    print("\nTraining session completed!")

    print(
        f"Best validation accuracy: {best_acc:.2f}%"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Checkpoints saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
