"""Model registry: lets new architectures be added without touching training code.

Usage:
    @register_model("my_model")
    class MyModel(nn.Module):
        ...

    model = build_model("my_model", in_channels=1, num_classes=10)
"""
from typing import Callable, Dict, Type

import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "<none registered>"
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](**kwargs)
