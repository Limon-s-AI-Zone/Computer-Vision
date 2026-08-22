import torch.nn as nn
import torchvision.models as tv_models

from .registry import register_model


@register_model("resnet18")
class ResNet18(nn.Module):
    """torchvision ResNet-18 adapted for arbitrary input channels / class count.

    Demonstrates that swapping to a "real" architecture only requires a new
    registered model class -- the data pipeline, trainer, and CLI are unchanged.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
