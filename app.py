"""Streamlit demo: serve any checkpoint trained by `python -m src.train`.

Run with:
    streamlit run app.py

Auto-detects available checkpoints in `checkpoints/`, infers which model
architecture and dataset each one was trained with from its filename
(`{model_name}_{dataset_name}_checkpoint.pth.tar`), and lets you classify an
image either by drawing it or uploading a file.
"""
import glob
import os

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from src.dataio import DATASET_REGISTRY, get_dataset_entry
from src.models import MODEL_REGISTRY, build_model
from src.utils.checkpoint import load_checkpoint

CHECKPOINT_DIR = "checkpoints"

CLASS_NAMES = {
    "fashion_mnist": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ],
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
}


def discover_checkpoints():
    """Parse `{model_name}_{dataset_name}_checkpoint.pth.tar` filenames back into
    (model_name, dataset_name) by matching known model names as a prefix -- this
    is the inverse of `format_paths()` in `src/utils/config.py`.
    """
    paths = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*_checkpoint.pth.tar")))
    model_names = sorted(MODEL_REGISTRY.keys(), key=len, reverse=True)

    entries = []
    for path in paths:
        base = os.path.basename(path)[: -len("_checkpoint.pth.tar")]
        model_name = next((m for m in model_names if base == m or base.startswith(m + "_")), None)
        dataset_name = base[len(model_name) + 1 :] if model_name else None
        entries.append(
            {
                "path": path,
                "label": base,
                "model_name": model_name,
                "dataset_name": dataset_name if dataset_name in DATASET_REGISTRY else None,
            }
        )
    return entries


def infer_num_classes_from_checkpoint(state_dict) -> int:
    """Fallback for datasets like `image_folder` whose registry num_classes is a
    placeholder (0): the last 2D ("Linear") weight tensor in the state dict is the
    final classification layer, whose output dim is the true class count.
    """
    last_linear_out = None
    for key, tensor in state_dict.items():
        if key.endswith(".weight") and tensor.dim() == 2:
            last_linear_out = tensor.shape[0]
    return last_linear_out


@st.cache_resource(show_spinner="Loading model...")
def load_model(checkpoint_path: str, model_name: str, dataset_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_checkpoint = torch.load(checkpoint_path, map_location=device)

    entry = get_dataset_entry(dataset_name)
    num_classes = entry.num_classes or infer_num_classes_from_checkpoint(raw_checkpoint["state_dict"])

    model = build_model(model_name, in_channels=entry.in_channels, num_classes=num_classes)
    model, _, _, _, _ = load_checkpoint(checkpoint_path, model, device=str(device))
    model.to(device).eval()
    return model, entry, num_classes, device


def preprocess(image: Image.Image, in_channels: int, image_size: int, invert: bool) -> torch.Tensor:
    mode = "L" if in_channels == 1 else "RGB"
    image = image.convert(mode)
    if invert and in_channels == 1:
        image = ImageOps.invert(image)
    image = image.resize((image_size, image_size))

    arr = np.array(image, dtype=np.float32) / 255.0
    if in_channels == 1:
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    else:
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def main():
    st.set_page_config(page_title="Image Classifier", page_icon="🔢", layout="centered")
    st.title("🔢 Image Classifier")
    st.caption("Serves checkpoints produced by this repo's `python -m src.train` pipeline.")

    checkpoints = discover_checkpoints()
    if not checkpoints:
        st.warning(
            f"No checkpoints found in `{CHECKPOINT_DIR}/`. Train a model first, e.g.:\n\n"
            "```bash\npython -m src.train --config configs/default.yaml\n```"
        )
        st.stop()

    st.sidebar.header("Model")
    selected_label = st.sidebar.selectbox("Checkpoint", [c["label"] for c in checkpoints])
    selected = next(c for c in checkpoints if c["label"] == selected_label)

    if selected["model_name"] and selected["dataset_name"]:
        model_name, dataset_name = selected["model_name"], selected["dataset_name"]
        st.sidebar.write(f"Architecture: **{model_name}**")
        st.sidebar.write(f"Trained on: **{dataset_name}**")
    else:
        st.sidebar.warning("Couldn't infer model/dataset from filename — please confirm:")
        model_name = st.sidebar.selectbox("Architecture", sorted(MODEL_REGISTRY.keys()))
        dataset_name = st.sidebar.selectbox("Dataset (defines input shape)", sorted(DATASET_REGISTRY.keys()))

    top_k = st.sidebar.slider("Show top-k predictions", min_value=1, max_value=10, value=3)

    model, entry, num_classes, device = load_model(selected["path"], model_name, dataset_name)
    class_names = CLASS_NAMES.get(dataset_name, [str(i) for i in range(num_classes)])

    draw_tab, upload_tab = st.tabs(["✏️ Draw", "📁 Upload"])
    image = None

    with draw_tab:
        try:
            from streamlit_drawable_canvas import st_canvas

            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=18,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            if canvas_result.image_data is not None and canvas_result.image_data[:, :, :3].sum() > 0:
                # White-on-black canvas already matches the MNIST-style convention -> no invert needed.
                image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("RGB")
        except ImportError:
            st.info(
                "Install `streamlit-drawable-canvas` to draw directly: "
                "`pip install streamlit-drawable-canvas`"
            )

    with upload_tab:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
        invert_upload = st.checkbox(
            "Invert colors (check this if it's a dark digit/object on a light background, e.g. a photo)",
            value=(entry.in_channels == 1),
        )
        if uploaded is not None:
            image = Image.open(uploaded)
            source = "upload"
        elif image is not None:
            source = "draw"
        else:
            source = None

    if image is None:
        st.info("Draw or upload an image to classify.")
        return

    invert = invert_upload if source == "upload" else False

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Input", width=150)

    tensor = preprocess(image, entry.in_channels, entry.image_size, invert=invert).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    topk_probs, topk_idx = torch.topk(probs, k=min(top_k, len(class_names)))

    with col2:
        pred_idx = int(topk_idx[0])
        st.metric("Prediction", class_names[pred_idx], f"{topk_probs[0]*100:.1f}% confidence")

    st.subheader(f"Top-{top_k} predictions")
    chart_data = {class_names[int(i)]: float(p) for p, i in zip(topk_probs, topk_idx)}
    st.bar_chart(chart_data)


if __name__ == "__main__":
    main()
