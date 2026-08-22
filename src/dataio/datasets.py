"""Concrete dataset registrations. Import this module (done in __init__.py) so the
decorators run and populate DATASET_REGISTRY.
"""
import os

import torchvision.datasets as tv_datasets

from .registry import register_dataset


@register_dataset("mnist", in_channels=1, image_size=28, num_classes=10)
def build_mnist(root, transform, download):
    train = tv_datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test = tv_datasets.MNIST(root=root, train=False, download=download, transform=transform)
    return train, test


@register_dataset("fashion_mnist", in_channels=1, image_size=28, num_classes=10)
def build_fashion_mnist(root, transform, download):
    train = tv_datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
    test = tv_datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)
    return train, test


@register_dataset("kmnist", in_channels=1, image_size=28, num_classes=10)
def build_kmnist(root, transform, download):
    train = tv_datasets.KMNIST(root=root, train=True, download=download, transform=transform)
    test = tv_datasets.KMNIST(root=root, train=False, download=download, transform=transform)
    return train, test


@register_dataset("cifar10", in_channels=3, image_size=32, num_classes=10)
def build_cifar10(root, transform, download):
    train = tv_datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    test = tv_datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
    return train, test


@register_dataset("image_folder", in_channels=3, image_size=224, num_classes=0)
def build_image_folder(root, transform, download):
    """Generic dataset for custom data laid out as:

        root/train/<class_name>/*.png
        root/test/<class_name>/*.png

    `num_classes` in the registry entry is a placeholder (0) since it depends on
    however many class subfolders exist under `root/train`; callers should read
    `len(train.classes)` if they need the true count.
    """
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")
    train = tv_datasets.ImageFolder(train_dir, transform=transform)
    test = tv_datasets.ImageFolder(test_dir, transform=transform)
    return train, test
