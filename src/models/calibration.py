"""
Temperature scaling: post-hoc calibration of classifier confidence, fit on the validation set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def fit_temperature(model, val_loader, device) -> float:
    model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)

    temperature = nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()

def apply_temperature(logits, temperature: float):
    return logits / temperature