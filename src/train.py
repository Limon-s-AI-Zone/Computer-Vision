"""Train an image classifier defined by a config file.

Examples:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --model.name deep_cnn
    python -m src.train --config configs/default.yaml --model.name resnet18 --train.num_epochs 10
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataio import get_dataloaders, get_dataset_entry
from src.engine.trainer import Trainer
from src.models import build_model
from src.utils.config import format_paths, load_config, merge_config
from src.utils.seed import set_seed
from src.utils.visualize import plot_accuracy, plot_confusion_matrix, plot_loss, plot_pr_curves, plot_roc_curves


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="section.key=value",
        help="Ad-hoc overrides, e.g. --set model.name=deep_cnn train.num_epochs=10",
    )
    return parser.parse_args()


def apply_dotted_overrides(cfg: dict, overrides: list) -> dict:
    override_dict = {}
    for item in overrides:
        key, value = item.split("=", 1)
        section, field = key.split(".", 1)
        override_dict.setdefault(section, {})[field] = _cast(value)
    return merge_config(cfg, override_dict)


def _cast(value: str):
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_dotted_overrides(cfg, args.set)
    cfg = format_paths(cfg)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = cfg["data"]["name"]
    dataset_entry = get_dataset_entry(dataset_name)

    train_loader, valid_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        valid_size=cfg["data"]["valid_size"],
        seed=cfg["seed"],
    )

    # model.in_channels / model.num_classes fall back to the dataset's registry
    # metadata, but can be overridden in config/CLI (e.g. for image_folder, where
    # num_classes depends on how many class folders exist under data.root).
    in_channels = cfg["model"].get("in_channels", dataset_entry.in_channels)
    num_classes = cfg["model"].get("num_classes") or dataset_entry.num_classes or len(train_loader.dataset.classes)

    model = build_model(cfg["model"]["name"], in_channels=in_channels, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        cfg["train"]["optimizer"], model.parameters(), cfg["train"]["learning_rate"], cfg["train"]["weight_decay"]
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_path=cfg["train"]["checkpoint_path"],
        num_classes=num_classes,
    )

    train_losses, valid_losses, train_accs, valid_accs = trainer.fit(
        train_loader,
        valid_loader,
        cfg["train"]["num_epochs"],
        save_checkpoint_enabled=cfg["train"]["save_checkpoint"],
        load_checkpoint_enabled=cfg["train"]["load_checkpoint"],
    )

    top_k = cfg["train"].get("top_k", 3)
    result = trainer.evaluate(test_loader, top_k=top_k)
    print(
        f"\nTest accuracy: {result.accuracy*100:.2f}% | precision: {result.precision:.4f} | "
        f"recall: {result.recall:.4f} | f1: {result.f1:.4f} | top-{top_k} accuracy: {result.top_k_accuracy*100:.2f}%"
    )

    plot_loss(train_losses, valid_losses, save_path=cfg["output"]["loss_plot_path"], show=False)
    plot_accuracy(train_accs, valid_accs, save_path=cfg["output"]["accuracy_plot_path"], show=False)
    plot_confusion_matrix(result.confusion_matrix, save_path=cfg["output"]["confusion_matrix_path"], show=False)
    plot_roc_curves(
        result.probs, result.labels, num_classes, save_path=cfg["output"]["roc_curve_path"], show=False
    )
    plot_pr_curves(
        result.probs, result.labels, num_classes, save_path=cfg["output"]["pr_curve_path"], show=False
    )

    results = {
        "model": cfg["model"]["name"],
        "dataset": dataset_name,
        "num_epochs": cfg["train"]["num_epochs"],
        "test_accuracy": result.accuracy,
        "test_precision": result.precision,
        "test_recall": result.recall,
        "test_f1": result.f1,
        "test_top_k_accuracy": result.top_k_accuracy,
        "top_k": top_k,
        "test_loss": result.loss,
        "classification_report": result.classification_report,
        "checkpoint_path": cfg["train"]["checkpoint_path"],
        "loss_plot_path": cfg["output"]["loss_plot_path"],
        "accuracy_plot_path": cfg["output"]["accuracy_plot_path"],
        "confusion_matrix_path": cfg["output"]["confusion_matrix_path"],
        "roc_curve_path": cfg["output"]["roc_curve_path"],
        "pr_curve_path": cfg["output"]["pr_curve_path"],
    }
    results_path = cfg["output"]["results_path"]
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results summary to {results_path}")


if __name__ == "__main__":
    main()
