import torch.nn as nn
from torchvision.models import resnet18


class CIFAR10ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 images.

    Input:
        3 x 32 x 32 RGB image

    Output:
        10 logits, one for each CIFAR-10 class
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Create a ResNet-18 without ImageNet pretrained weights.
        self.model = resnet18(weights=None)

        # CIFAR-10 images are small (32x32), so use
        # a smaller first convolution and avoid early downsampling.
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.model.maxpool = nn.Identity()

        # CIFAR-10 has 10 classes.
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes,
        )

    def forward(self, x):
        return self.model(x)