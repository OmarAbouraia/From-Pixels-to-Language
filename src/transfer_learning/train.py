import argparse
import csv
import torch

from pathlib import Path

from src.transfer_learning.data import (
    create_transfer_learning_loaders,
)

from src.transfer_learning.model import (
    create_resnet18,
)

from src.transfer_learning.engine import (
    train_one_epoch,
    validate_one_epoch,
)


def run_experiment(
    mode,
    epochs=10,
    batch_size=32,
    learning_rate=None,
    weight_decay=0.0,
    num_workers=0,
    output_dir=None,
):
    """
    Run a complete Transfer Learning experiment.

    Args:
        mode: 'feature_extraction' or 'fine_tuning'.
        epochs: Number of training epochs.
        batch_size: DataLoader batch size.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularization coefficient.
        num_workers: DataLoader workers.
        output_dir: Directory for saving experiment outputs.
    """

    # 1. Configure experiment
    if learning_rate is None:
        learning_rate = (
            0.001 if mode == "feature_extraction" else 0.0001
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("TRANSFER LEARNING EXPERIMENT")
    print("=" * 60)

    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")

    if device.type == "cpu":
        print("\nWARNING: CUDA is unavailable. Training will use CPU.")

    # 2. Load DataLoaders
    print("\nLoading dataset...")

    train_loader, val_loader = create_transfer_learning_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # 3. Initialize model
    print("\nInitializing ResNet18...")

    model = create_resnet18(
        mode=mode,
        pretrained=True,
    )

    model = model.to(device)

    # 4. Initialize loss function
    criterion = torch.nn.CrossEntropyLoss()

    # 5. Initialize optimizer
    trainable_parameters = filter(
        lambda parameter: parameter.requires_grad,
        model.parameters(),
    )

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # 6. Prepare output directory
    if output_dir is None:

        project_root = Path(__file__).resolve().parents[2]

        output_dir = (
            project_root / "outputs" / "transfer_learning" / mode
        )

    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pth"
    last_checkpoint_path = output_dir / "last_checkpoint.pth"
    history_path = output_dir / "training_history.csv"

    # 7. Training loop
    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):

        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print("-" * 40)

        # Training
        train_results = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            mode=mode,
        )

        # Validation
        val_results = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        train_loss = train_results["loss"]
        train_accuracy = train_results["accuracy"]

        val_loss = val_results["loss"]
        val_accuracy = val_results["accuracy"]

        # Display results
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        # Record epoch results
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        # Check whether the model improved
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

        # Prepare checkpoint information
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "mode": mode,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        }

        # Save best model only when validation loss improves
        if is_best:
            torch.save(
                checkpoint,
                checkpoint_path,
            )

            print("Best model checkpoint saved!")

        # Always save the latest training state
        torch.save(
            checkpoint,
            last_checkpoint_path,
        )

        print("Latest checkpoint saved!")

        # Save training history after every epoch
        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=history[0].keys(),
            )

            writer.writeheader()
            writer.writerows(history)

        print("Training history updated!")

    # 8. Experiment summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)

    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Checkpoint: {checkpoint_path}")
    print(f"Latest Checkpoint: {last_checkpoint_path}")
    print(f"Training History: {history_path}")

    return history


def main():

    parser = argparse.ArgumentParser(
        description="ResNet18 Transfer Learning Experiments"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["feature_extraction", "fine_tuning"],
        default="feature_extraction",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints and training results.",
    )

    args = parser.parse_args()

    run_experiment(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()