from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 64,
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
):
    """
    Create CIFAR-10 training and test DataLoaders.

    Training data uses random augmentation.
    Test data uses only deterministic preprocessing.
    """

    data_path = Path(data_dir)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=test_transform,
    )

    if max_train_samples is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(max_train_samples, len(train_dataset))),
        )

    if max_test_samples is not None:
        test_dataset = Subset(
            test_dataset,
            range(min(max_test_samples, len(test_dataset))),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader