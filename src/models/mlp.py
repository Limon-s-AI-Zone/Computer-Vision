import torch.nn as nn

from .registry import register_model


@register_model("mlp")
class MLP(nn.Module):
    """Fully-connected baseline for quick experiments / architecture comparisons."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, image_size: int = 28, hidden_dim: int = 256):
        super().__init__()
        input_dim = in_channels * image_size * image_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
