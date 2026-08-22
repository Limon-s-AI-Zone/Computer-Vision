import os
from typing import Tuple

import numpy as np
import torch


def save_checkpoint(filepath: str, model, optimizer, val_acc: float, val_loss: float, epoch: int) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": val_loss,
        "acc": val_acc,
    }
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model, optimizer=None, device: str = "cpu") -> Tuple[object, object, int, float, float]:
    """Loads a checkpoint in-place if it exists. Returns (model, optimizer, start_epoch, best_acc, valid_loss_min)."""
    start_epoch = 1
    best_acc = 0.0
    valid_loss_min = np.inf

    if filepath and os.path.isfile(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 1)
        best_acc = checkpoint.get("acc", 0.0)
        valid_loss_min = checkpoint.get("loss", np.inf)
        print(f"=> loaded checkpoint '{filepath}' (epoch {start_epoch})")
    else:
        print(f"=> no checkpoint found at '{filepath}', starting from scratch")

    return model, optimizer, start_epoch, best_acc, valid_loss_min
