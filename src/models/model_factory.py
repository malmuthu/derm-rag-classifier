"""
Model factory: build either a ResNet50 or ViT-B/16 backbone, adapted for 7-class HAM10000 classification.
"""

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

NUM_CLASSES = 7

def build_model(backbone: str, dropout: float = 0.3) -> nn.Module:
    if backbone == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, NUM_CLASSES)
        )

    elif backbone == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, NUM_CLASSES)
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model

def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_head_only(model: nn.Module, backbone: str) -> None:
    freeze_all(model)
    if backbone == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif backbone == "vit_b_16":
        for param in model.heads.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

def unfreeze_partial(model: nn.Module, backbone: str) -> None:
    freeze_all(model)
    if backbone == "resnet50":
        for layer in [model.layer3, model.layer4, model.fc]:
            for param in layer.parameters():
                param.requires_grad = True
    elif backbone == "vit_b_16":
        for block in model.encoder.layers[8:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.heads.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown backbone {backbone}")

def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total