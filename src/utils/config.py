import copy
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge override into a copy of base (override wins). Used for CLI overrides."""
    merged = copy.deepcopy(base)
    return _deep_update(merged, override)


def format_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Substitutes {model_name}/{dataset_name} in path templates so each
    model+dataset combination gets its own checkpoint/plots/results files.
    """
    fmt_kwargs = {"model_name": cfg["model"]["name"], "dataset_name": cfg["data"]["name"]}
    cfg["train"]["checkpoint_path"] = cfg["train"]["checkpoint_path"].format(**fmt_kwargs)
    for key in (
        "loss_plot_path",
        "accuracy_plot_path",
        "confusion_matrix_path",
        "roc_curve_path",
        "pr_curve_path",
        "results_path",
    ):
        cfg["output"][key] = cfg["output"][key].format(**fmt_kwargs)
    return cfg
