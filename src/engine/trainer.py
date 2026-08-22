from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.engine.metrics import (
    calculate_accuracy,
    classification_report,
    confusion_matrix,
    precision_recall_f1,
    top_k_accuracy,
)
from src.utils.checkpoint import load_checkpoint, save_checkpoint


@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    top_k_accuracy: float
    loss: float
    confusion_matrix: torch.Tensor
    classification_report: Dict
    probs: torch.Tensor
    labels: torch.Tensor


class Trainer:
    """Model-agnostic training/eval loop.

    Works with any nn.Module produced by `build_model`, so switching
    architectures only means changing `model.name` in the config.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_path: str,
        num_classes: int = 10,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes

    def load_if_available(self, enabled: bool = True):
        filepath = self.checkpoint_path if enabled else ""
        self.model, self.optimizer, start_epoch, best_acc, valid_loss_min = load_checkpoint(
            filepath, self.model, self.optimizer, device=str(self.device)
        )
        return start_epoch, best_acc, valid_loss_min

    def train_one_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for data, target in tqdm(train_loader, leave=False):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, correct_batch, total_batch = calculate_accuracy(output, target)
            correct += correct_batch
            total += total_batch

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, top_k: int = 3) -> EvalResult:
        self.model.eval()
        all_outputs, all_labels = [], []
        running_loss = 0.0

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)

            running_loss += loss.item()
            all_outputs.append(output)
            all_labels.append(target)

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        accuracy, _, _ = calculate_accuracy(all_outputs, all_labels)
        precision, recall, f1 = precision_recall_f1(all_outputs, all_labels, self.num_classes)
        topk_acc = top_k_accuracy(all_outputs, all_labels, k=top_k)
        cm = confusion_matrix(all_outputs, all_labels, self.num_classes)
        report = classification_report(all_outputs, all_labels, self.num_classes)
        probs = torch.softmax(all_outputs, dim=1)
        avg_loss = running_loss / len(data_loader)

        return EvalResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            top_k_accuracy=topk_acc,
            loss=avg_loss,
            confusion_matrix=cm,
            classification_report=report,
            probs=probs,
            labels=all_labels,
        )

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int,
        save_checkpoint_enabled: bool = True,
        load_checkpoint_enabled: bool = True,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        start_epoch, _, valid_loss_min = self.load_if_available(enabled=load_checkpoint_enabled)

        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []

        for epoch in range(start_epoch, num_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            result = self.evaluate(valid_loader)
            valid_acc, valid_loss = result.accuracy, result.loss

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)

            epoch_len = len(str(num_epochs))
            print(
                f"[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] "
                f"train_loss: {train_loss:.5f} train_acc: {train_acc:.2f}% "
                f"valid_loss: {valid_loss:.5f} valid_acc: {valid_acc*100:.2f}% "
                f"precision: {result.precision:.4f} recall: {result.recall:.4f} f1: {result.f1:.4f}"
            )

            if save_checkpoint_enabled and valid_loss <= valid_loss_min:
                print(f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...")
                save_checkpoint(self.checkpoint_path, self.model, self.optimizer, valid_acc, valid_loss, epoch)
                valid_loss_min = valid_loss

        return train_losses, valid_losses, train_accs, valid_accs
