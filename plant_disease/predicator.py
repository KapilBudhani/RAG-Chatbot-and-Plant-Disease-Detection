"""Lazy-loaded plant disease classifier inference."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from PIL import Image

from app.config import PLANT_MODEL_PATH


def model_exists() -> bool:
    return PLANT_MODEL_PATH.exists()


def prettify_label(label: str) -> dict:
    raw = str(label).strip()
    crop_prefixes = [
        "Cherry_(including_sour)", "Corn_(maize)", "Pepper,_bell",
        "Blueberry", "Raspberry", "Strawberry", "Apple", "Cherry",
        "Corn", "Grape", "Orange", "Peach", "Pepper", "Potato",
        "Soybean", "Squash", "Tomato",
    ]
    crop_raw = ""
    condition_raw = ""
    for prefix in sorted(crop_prefixes, key=len, reverse=True):
        if raw.lower().startswith(prefix.lower()):
            suffix = raw[len(prefix):]
            if suffix.startswith(("___", "__", "_", " ")):
                crop_raw = prefix
                condition_raw = suffix.lstrip("_ ")
                break
    if not crop_raw:
        for separator in ("___", "__"):
            if separator in raw:
                crop_raw, condition_raw = raw.split(separator, 1)
                break
    if not crop_raw:
        if "_" in raw:
            crop_raw, condition_raw = raw.split("_", 1)
        else:
            crop_raw, condition_raw = raw, "Unknown condition"
    crop = crop_raw.replace("_", " ").replace(",", "").strip().title()
    condition = condition_raw.replace("_", " ").replace("(", "").replace(")", "").strip().title()
    healthy = "healthy" in condition.lower()
    if healthy:
        condition = "Healthy"
    return {
        "class_name": raw,
        "display_name": f"{crop or 'Unknown crop'} — {condition or 'Unknown condition'}",
        "crop": crop or "Unknown crop",
        "condition": condition or "Unknown condition",
        "is_healthy": healthy,
    }


@lru_cache(maxsize=2)
def _load_model_cached(model_path_text: str, modified_time: float):
    """Cache the model and preprocessing pipeline per file version."""
    import torch
    from torchvision import transforms
    from plant_disease.model_utils import create_model

    model_path = Path(model_path_text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    image_size = int(checkpoint.get("image_size", 224))
    model = create_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            checkpoint.get("normalization_mean", [0.485, 0.456, 0.406]),
            checkpoint.get("normalization_std", [0.229, 0.224, 0.225]),
        ),
    ])
    return model, classes, device, transform


def predict_image(image: Image.Image, top_k: int = 3) -> dict:
    """Run inference. PyTorch/torchvision imports happen only when this function is called."""
    if not model_exists():
        raise FileNotFoundError(
            f"Trained plant disease model not found at {PLANT_MODEL_PATH}. "
            "Download the Kaggle dataset and run plant_disease/train.py first."
        )
    import torch

    model, classes, device, transform = _load_model_cached(
        str(PLANT_MODEL_PATH), PLANT_MODEL_PATH.stat().st_mtime
    )
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor)[0], dim=0)
        values, indices = torch.topk(probabilities, k=min(max(1, top_k), len(classes)))

    top_predictions = []
    for prob, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
        details = prettify_label(classes[idx])
        top_predictions.append({**details, "confidence": float(prob)})
    primary = top_predictions[0]
    return {
        **primary,
        "confidence": primary["confidence"],
        "top_predictions": top_predictions,
        "confidence_is_calibrated": False,
        "model_path": str(PLANT_MODEL_PATH),
    }
