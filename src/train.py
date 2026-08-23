from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import CIFAR10ResNet18

import argparse
import json


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

    model = CIFAR10ResNet18().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = Path(
        config["output"]["checkpoint_dir"]
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = checkpoint_dir / config["output"]["checkpoint_name"]

    metrics_path = checkpoint_dir / config["output"]["metrics_file"]

    best_accuracy = -1.0
    epochs_without_improvement = 0

    # Start a fresh metrics file for this training run.
    metrics_path.write_text("", encoding="utf-8")

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

        metrics = {
            "epoch": epoch + 1,
            "train_loss": average_loss,
            "test_accuracy": accuracy,
        }

        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")

        print(json.dumps(metrics))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                f"Best model checkpoint saved to: "
                f"{checkpoint_path}"
            )

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping triggered after "
                    f"{epoch + 1} epochs."
                )
                break

    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Metrics saved to: {metrics_path}")


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