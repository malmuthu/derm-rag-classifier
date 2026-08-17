"""
Extract image embeddings from a trained ViT-B/16 classifier for use in the
retrieval layer. Embeddings are the 768-dimension feature vector immediately
preceding the final classification layer.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data"))
sys.path.append(str(PROJECT_ROOT / "src" / "models"))

from dataset import eval_transform
from model_factory import build_model

def load_embedding_model(checkpoint_path, device) -> nn.Module:
    model = build_model("vit_b_16")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # feature vector instead of 7-class logits
    model.heads.head = nn.Identity()

    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def get_embedding(model, image, device):
    tensor = eval_transform(image).unsqueeze(0).to(device)
    embedding = model(tensor)
    return embedding.squeeze(0).cpu().numpy()