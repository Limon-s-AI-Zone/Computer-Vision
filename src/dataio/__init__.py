from .registry import DATASET_REGISTRY, get_dataset_entry, register_dataset

# Import dataset registrations so their @register_dataset decorators run.
from . import datasets  # noqa: E402,F401
from .dataset import get_dataloaders  # noqa: E402

__all__ = ["DATASET_REGISTRY", "register_dataset", "get_dataset_entry", "get_dataloaders"]
