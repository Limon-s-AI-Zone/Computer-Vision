from typing import Optional, Tuple

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .registry import DatasetEntry, get_dataset_entry


def get_dataloaders(
    dataset_name: str = "mnist",
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    valid_size: float = 0.2,
    seed: int = 42,
    download: bool = True,
    transform: Optional[transforms.Compose] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/valid/test DataLoaders for any registered dataset.

    Add support for a new dataset by registering it in `src/data/datasets.py`
    (see `register_dataset`) -- this function and the trainer never need to change.
    """
    entry: DatasetEntry = get_dataset_entry(dataset_name)

    if transform is None:
        transform = transforms.ToTensor()

    train_data, test_data = entry.factory(root=root, transform=transform, download=download)

    num_train = len(train_data)
    indices = list(range(num_train))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    common_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, **common_kwargs)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, **common_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **common_kwargs)

    return train_loader, valid_loader, test_loader
