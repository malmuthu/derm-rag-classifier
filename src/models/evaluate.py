"""
Evaluate a trained checkpoint on the held-out test set.
Run: python src/models/evaluate.py --config configs/resnet50.yaml
"""
import sys
import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data"))
sys.path.append(str(PROJECT_ROOT / "src" / "utils"))

from dataset import get_dataloaders, CLASSES
from model_factory import build_model
from config import load_config
from train import get_device


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def apply_mel_thresold(probs: np.ndarray, threshold: float) -> np.ndarray:
    mel_idx = CLASSES.index("mel")
    preds = probs.argmax(axis=1)

    override_mask = probs[:, mel_idx] > threshold
    preds[override_mask] = mel_idx

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mel-threshold", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    _, _, test_loader = get_dataloaders(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    model = build_model(config["backbone"]).to(device)
    checkpoint_path = Path(config["checkpoint_dir"]) / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    labels, preds, probs = get_predictions(model, test_loader, device)

    if args.mel_threshold is not None:
        preds = apply_mel_thresold(probs, args.mel_threshold)
        print(f"Applied mel threshold: {args.mel_threshold}")

    print(f"\n=== {config['backbone']} — Test Set Results ===\n")
    print(classification_report(labels, preds, target_names=CLASSES, digits=3))

    print("Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(labels, preds)
    print("       " + "  ".join(f"{c:>6s}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>6s} " + "  ".join(f"{v:>6d}" for v in row))


if __name__ == "__main__":
    main()