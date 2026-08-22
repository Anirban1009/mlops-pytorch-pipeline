from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import CIFAR10ResNet18

import argparse


def load_config(config_path: str):
    """Load training configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate(model, data_loader, device):
    """Evaluate model accuracy on a dataset."""
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return 100.0 * correct / total


def train(config_path: str):
    config = load_config(config_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        max_train_samples=config["data"]["max_train_samples"],
        max_test_samples=config["data"]["max_test_samples"],
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = CIFAR10ResNet18()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        accuracy = evaluate(
            model,
            test_loader,
            device,
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.4f} "
            f"Test Accuracy: {accuracy:.2f}%"
        )

    checkpoint_dir = Path(
        config["output"]["checkpoint_dir"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = checkpoint_dir / "model.pt"

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    print(f"Model checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 ResNet-18 model"
    )

    parser.add_argument(
        "--config",
        default="configs/training_config.yaml",
        help="Path to training configuration YAML file",
    )

    args = parser.parse_args()

    train(args.config)