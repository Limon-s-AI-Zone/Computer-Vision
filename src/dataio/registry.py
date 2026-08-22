"""Dataset registry: lets new datasets be added without touching the dataloader code.

Each entry is a factory `(root, transform, download) -> (train_dataset, test_dataset)`
plus metadata (in_channels, image_size, num_classes) so models/configs can stay in sync
with whichever dataset is active.

Usage:
    @register_dataset("my_dataset", in_channels=3, image_size=32, num_classes=10)
    def build_my_dataset(root, transform, download):
        train = MyTrainDataset(root, transform=transform, download=download)
        test = MyTestDataset(root, transform=transform, download=download)
        return train, test
"""
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from torch.utils.data import Dataset

DatasetFactory = Callable[..., Tuple[Dataset, Dataset]]


@dataclass
class DatasetEntry:
    factory: DatasetFactory
    in_channels: int
    image_size: int
    num_classes: int


DATASET_REGISTRY: Dict[str, DatasetEntry] = {}


def register_dataset(name: str, in_channels: int, image_size: int, num_classes: int):
    def decorator(factory: DatasetFactory) -> DatasetFactory:
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered.")
        DATASET_REGISTRY[name] = DatasetEntry(
            factory=factory, in_channels=in_channels, image_size=image_size, num_classes=num_classes
        )
        return factory

    return decorator


def get_dataset_entry(name: str) -> DatasetEntry:
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY)) or "<none registered>"
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return DATASET_REGISTRY[name]
