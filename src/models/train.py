"""
Training script - backbone-agnostic, config-driven
Run: python src/models/train.py --config configs/resnet50.yaml
"""

import argparse
import sys
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data"))
sys.path.append(str(PROJECT_ROOT / "src" / "utils"))

from dataset import get_dataloaders, CLASSES
from model_factory import build_model, unfreeze_head_only, unfreeze_partial, count_trainable_params
from losses import get_loss_fn
from config import load_config

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    mel_idx = CLASSES.index("mel")
    return{
        "loss": total_loss / len(loader.dataset),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "mel_recall": recall_score(all_labels, all_preds, labels=[mel_idx], average="macro"),
    }

def run_phase(model, train_loader, val_loader, loss_fn, device, epochs, lr,
              weight_decay, checkpoint_dir, phase_name, best_f1_so_far):
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    #scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = best_f1_so_far
    #epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        #current_lr = scheduler.get_last_lr()[0]

        print(
            f"[{phase_name}] epoch {epoch} / {epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_mel_recall={val_metrics['mel_recall']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            print(f"  -> new best (macro_f1={best_f1:.4f}), checkpoint saved")


        #if epochs_without_improvement >= early_stopping_patience:
         #   print(f"  -> no improvement for {early_stopping_patience} epochs, stopping early")
        #    break

    return best_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config["seed"])

    device = get_device()
    print(f"Using device: {device}")

    backbone = config["backbone"]
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_weighted_sampler=config.get("use_weighted_sampler", False),
        sampler_softening=config.get("sampler_softening", "sqrt"),
    )

    model = build_model(backbone).to(device)

    train_csv = PROJECT_ROOT / "data" / "processed" / "train.csv"
    #if config.get("use_weighted_sampler", False):
    #    loss_fn = nn.CrossEntropyLoss()
    #else:
    loss_fn = get_loss_fn(train_csv, device)

    # Phase 1
    unfreeze_head_only(model, backbone)
    trainable, total = count_trainable_params(model)
    print(f"Phase 1 - trainable params: {trainable:,} / {total:,}")

    best_f1 = run_phase(
        model, train_loader, val_loader, loss_fn, device,
        epochs=config["phase1"]["epochs"],
        lr=config["phase1"]["learning_rate"],
        weight_decay=config["phase1"]["weight_decay"],
        checkpoint_dir=config["checkpoint_dir"],
        phase_name="phase1",
        best_f1_so_far=0.01,
    )

    # Phase 2
    unfreeze_partial(model, backbone)
    trainable, total = count_trainable_params(model)
    print(f"Phase 2 - trainable params: {trainable:,} / {total:,}")

    best_f1 = run_phase(
        model, train_loader, val_loader, loss_fn, device,
        epochs=config["phase2"]["epochs"],
        lr=config["phase2"]["learning_rate"],
        weight_decay=config["phase2"]["weight_decay"],
        checkpoint_dir=config["checkpoint_dir"],
        phase_name="phase2",
        best_f1_so_far=best_f1,
    )

    print(f"\nTraining complete. Best val macro f1: {best_f1:.4f}")
    print(f"Best checkpoint: {config['checkpoint_dir']}/best_model.pt")

if __name__ == "__main__":
    main()