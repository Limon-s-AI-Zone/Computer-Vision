from typing import Dict, List, Optional, Tuple

import torch


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, int, int]:
    _, preds = torch.max(outputs, 1)
    correct = torch.sum(preds == labels).item()
    total = labels.size(0)
    return correct / total, correct, total


def precision_recall_f1(outputs: torch.Tensor, labels: torch.Tensor, num_classes: int = 10):
    _, preds = torch.max(outputs, 1)

    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)

    for i in range(num_classes):
        TP[i] = torch.sum((preds == i) & (labels == i)).item()
        FP[i] = torch.sum((preds == i) & (labels != i)).item()
        FN[i] = torch.sum((preds != i) & (labels == i)).item()

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision.mean().item(), recall.mean().item(), f1.mean().item()


def confusion_matrix(outputs: torch.Tensor, labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    _, preds = torch.max(outputs, 1)
    cm = torch.zeros(num_classes, num_classes)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def top_k_accuracy(outputs: torch.Tensor, labels: torch.Tensor, k: int = 3) -> float:
    """Fraction of samples where the true label is among the top-k predicted classes."""
    k = min(k, outputs.shape[1])
    topk_preds = outputs.topk(k, dim=1).indices
    correct = topk_preds.eq(labels.view(-1, 1)).any(dim=1)
    return correct.float().mean().item()


def classification_report(
    outputs: torch.Tensor, labels: torch.Tensor, num_classes: int = 10, class_names: Optional[List[str]] = None
) -> Dict:
    """Per-class precision/recall/f1/support, plus macro/weighted averages.

    Uses scikit-learn's classification_report under the hood since it already
    handles zero-support classes and averaging correctly.
    """
    from sklearn.metrics import classification_report as sk_classification_report

    _, preds = torch.max(outputs, 1)
    target_names = class_names or [str(i) for i in range(num_classes)]

    return sk_classification_report(
        labels.cpu().numpy(),
        preds.cpu().numpy(),
        labels=list(range(num_classes)),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
