import torch

from src.model import CIFAR10ResNet18


def test_model_output_shape():
    model = CIFAR10ResNet18()
    model.eval()

    inputs = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 10)