import torch

from src.engine.metrics import (
    calculate_accuracy,
    classification_report,
    confusion_matrix,
    precision_recall_f1,
    top_k_accuracy,
)


def test_calculate_accuracy_perfect():
    outputs = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1])
    acc, correct, total = calculate_accuracy(outputs, labels)
    assert acc == 1.0
    assert correct == 2
    assert total == 2


def test_precision_recall_f1_perfect():
    outputs = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1])
    precision, recall, f1 = precision_recall_f1(outputs, labels, num_classes=2)
    assert precision > 0.99
    assert recall > 0.99
    assert f1 > 0.99


def test_confusion_matrix_shape():
    outputs = torch.randn(10, 10)
    labels = torch.randint(0, 10, (10,))
    cm = confusion_matrix(outputs, labels, num_classes=10)
    assert cm.shape == (10, 10)
    assert cm.sum().item() == 10


def test_top_k_accuracy_perfect_at_k1():
    outputs = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1])
    assert top_k_accuracy(outputs, labels, k=1) == 1.0


def test_top_k_accuracy_relaxes_with_larger_k():
    # true label is always second-highest logit -> top-1 wrong, top-2 correct
    outputs = torch.tensor([[10.0, 9.0, 0.0], [0.0, 10.0, 9.0]])
    labels = torch.tensor([1, 2])
    assert top_k_accuracy(outputs, labels, k=1) == 0.0
    assert top_k_accuracy(outputs, labels, k=2) == 1.0


def test_classification_report_has_per_class_and_macro_avg():
    outputs = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
    labels = torch.tensor([0, 1, 0])
    report = classification_report(outputs, labels, num_classes=2, class_names=["a", "b"])
    assert set(report.keys()) >= {"a", "b", "macro avg", "weighted avg", "accuracy"}
    assert report["a"]["support"] == 2
    assert report["b"]["support"] == 1
