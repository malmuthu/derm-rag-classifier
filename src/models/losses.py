"""
Class-weighted loss for handling HAM10000's class imbalance
"""

import pandas as pd
import sys
import torch
import torch.nn as nn
from dataset import CLASSES
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT / "src" / "data"))

def compute_class_weights(train_csv_path) -> torch.Tensor:
    df = pd.read_csv(train_csv_path)
    counts = df["dx"].value_counts()

    total = len(df)
    num_classes = len(CLASSES)

    weights = []
    for cls in CLASSES:
        count = counts.get(cls, 0)
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)

def get_loss_fn(train_csv_path, device) -> nn.Module:
    weights = compute_class_weights(train_csv_path).to(device)
    return nn.CrossEntropyLoss(weight=weights)
