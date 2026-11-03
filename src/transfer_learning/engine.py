import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device, mode):
    """Train the model for one epoch.

    Args:
        model: ResNet18 model.
        dataloader: Training DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimizer (e.g. Adam).
        device: CPU or CUDA device.
        mode: 'feature_extraction' or 'fine_tuning'.

    Returns:
        Dictionary containing average loss and accuracy.
    """
    model.train()

    # Keep the frozen backbone's BatchNorm layers fixed
    if mode == "feature_extraction":
        model.eval()
        model.fc.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return {
        "loss": average_loss,
        "accuracy": accuracy,
    }


def validate_one_epoch(model, dataloader, criterion, device):
    """Validate the model for one epoch.

    Returns:
        Dictionary containing average loss and accuracy.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return {
        "loss": average_loss,
        "accuracy": accuracy,
    }
