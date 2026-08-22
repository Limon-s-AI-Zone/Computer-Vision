"""Run inference with a trained checkpoint on a single image file.

Example:
    python -m src.predict --config configs/default.yaml --image path/to/digit.png
"""
import argparse

import torch
from PIL import Image
from torchvision import transforms

from src.dataio import get_dataset_entry
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import format_paths, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to a grayscale digit image.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = format_paths(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_entry = get_dataset_entry(cfg["data"]["name"])
    in_channels = cfg["model"].get("in_channels", dataset_entry.in_channels)
    num_classes = cfg["model"].get("num_classes") or dataset_entry.num_classes

    model = build_model(cfg["model"]["name"], in_channels=in_channels, num_classes=num_classes)
    model, _, _, _, _ = load_checkpoint(cfg["train"]["checkpoint_path"], model, device=str(device))
    model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=in_channels) if in_channels != 3 else (lambda img: img.convert("RGB")),
            transforms.Resize((dataset_entry.image_size, dataset_entry.image_size)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(args.image)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())

    print(f"Predicted class: {pred}")
    for i, p in enumerate(probs.tolist()):
        print(f"  {i}: {p:.4f}")


if __name__ == "__main__":
    main()
