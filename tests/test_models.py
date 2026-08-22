import torch

from src.models import MODEL_REGISTRY, build_model


def test_all_models_registered():
    assert {"simple_cnn", "deep_cnn", "mlp", "resnet18"} <= set(MODEL_REGISTRY.keys())


def test_all_models_forward_pass():
    x = torch.randn(2, 1, 28, 28)
    for name in MODEL_REGISTRY:
        model = build_model(name, in_channels=1, num_classes=10)
        out = model(x)
        assert out.shape == (2, 10), f"{name} produced wrong output shape: {out.shape}"
