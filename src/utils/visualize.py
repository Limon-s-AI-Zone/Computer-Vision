import itertools
import os
from typing import List, Optional

import numpy as np
import torch


def plot_dataset(data_loader):
    import matplotlib.pyplot as plt

    images, labels = next(iter(data_loader))
    images = images.numpy()

    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(min(20, len(images))):
        ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap="gray")
        ax.set_title(str(labels[idx].item()))
    plt.show()


def plot_confusion_matrix(
    cm: torch.Tensor, class_names: List[str] = None, save_path: str = None, show: bool = True
):
    import matplotlib.pyplot as plt

    cm_np = cm.numpy()
    class_names = class_names or [str(i) for i in range(cm_np.shape[0])]

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm_np, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm_np.max() / 2
    for i, j in itertools.product(range(cm_np.shape[0]), range(cm_np.shape[1])):
        plt.text(
            j, i, format(cm_np[i, j], ".0f"),
            horizontalalignment="center",
            color="white" if cm_np[i, j] > thresh else "black",
        )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_loss(train_loss: List[float], valid_loss: List[float], save_path: str = None, show: bool = True):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label="Validation Loss")

    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle="--", color="r", label="Early Stopping Checkpoint")

    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xlim(0, len(train_loss) + 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_accuracy(train_acc: List[float], valid_acc: List[float], save_path: str = None, show: bool = True):
    """Train/valid accuracy per epoch. `train_acc` is expected as a percentage (0-100),
    `valid_acc` as a fraction (0-1) -- matching what `Trainer.fit` returns -- and both
    are plotted on a common 0-100% scale.
    """
    import matplotlib.pyplot as plt

    valid_acc_pct = [v * 100 for v in valid_acc]

    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label="Training Accuracy")
    plt.plot(range(1, len(valid_acc_pct) + 1), valid_acc_pct, label="Validation Accuracy")

    maxpos = valid_acc_pct.index(max(valid_acc_pct)) + 1
    plt.axvline(maxpos, linestyle="--", color="g", label="Best Validation Accuracy")

    plt.xlabel("epochs")
    plt.ylabel("accuracy (%)")
    plt.xlim(0, len(train_acc) + 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_roc_curves(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    save_path: str = None,
    show: bool = True,
):
    """One-vs-rest ROC curve + AUC per class."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    class_names = class_names or [str(i) for i in range(num_classes)]
    y_true = label_binarize(labels.cpu().numpy(), classes=list(range(num_classes)))
    y_score = probs.cpu().numpy()

    fig = plt.figure(figsize=(8, 7))
    aucs = []
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (one-vs-rest, macro-avg AUC={np.mean(aucs):.3f})")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_pr_curves(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    save_path: str = None,
    show: bool = True,
):
    """One-vs-rest Precision-Recall curve + average precision per class."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.preprocessing import label_binarize

    class_names = class_names or [str(i) for i in range(num_classes)]
    y_true = label_binarize(labels.cpu().numpy(), classes=list(range(num_classes)))
    y_score = probs.cpu().numpy()

    fig = plt.figure(figsize=(8, 7))
    aps = []
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        ap = average_precision_score(y_true[:, i], y_score[:, i])
        aps.append(ap)
        plt.plot(recall, precision, label=f"{class_names[i]} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves (one-vs-rest, macro-avg AP={np.mean(aps):.3f})")
    plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
