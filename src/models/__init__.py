from .registry import MODEL_REGISTRY, register_model, build_model

# Import model modules so their @register_model decorators run and populate the registry.
from . import cnn  # noqa: E402,F401
from . import mlp  # noqa: E402,F401
from . import resnet  # noqa: E402,F401

__all__ = ["MODEL_REGISTRY", "register_model", "build_model"]
