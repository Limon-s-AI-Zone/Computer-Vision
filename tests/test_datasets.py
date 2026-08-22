from src.dataio import DATASET_REGISTRY, get_dataset_entry


def test_expected_datasets_registered():
    assert {"mnist", "fashion_mnist", "kmnist", "cifar10", "image_folder"} <= set(DATASET_REGISTRY.keys())


def test_get_dataset_entry_metadata():
    entry = get_dataset_entry("cifar10")
    assert entry.in_channels == 3
    assert entry.image_size == 32
    assert entry.num_classes == 10


def test_unknown_dataset_raises():
    try:
        get_dataset_entry("not_a_real_dataset")
        assert False, "expected ValueError"
    except ValueError:
        pass
