"""
Build the retrieval case database: embed all training images,
store embeddings + metadata for later similarity search.
Run: python src/retrieval/build_case_ddb.py
"""

import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data"))
sys.path.append(str(PROJECT_ROOT / "src" / "models"))
sys.path.append(str(PROJECT_ROOT / "src" / "utils"))

from embeddings import load_embedding_model, get_embedding
from config import load_config

IMG_DIR_1 = PROJECT_ROOT / "data" / "raw" / "HAM10000_images_part_1"
IMG_DIR_2 = PROJECT_ROOT / "data" / "raw" / "HAM10000_images_part_2"

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    retrieval_config = load_config(PROJECT_ROOT / "configs" / "retrieval.yaml")
    checkpoint_path = PROJECT_ROOT / retrieval_config["embedding_checkpoint"]
    model = load_embedding_model(checkpoint_path, device)

    train_csv = PROJECT_ROOT / "data" / "processed" / "train.csv"
    df = pd.read_csv(train_csv)

    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding case database"):
        img_path = IMG_DIR_1 / f"{row['image_id']}.jpg"
        if not img_path.exists():
            img_path = IMG_DIR_2 / f"{row['image_id']}.jpg"

        image = Image.open(img_path).convert("RGB")
        embedding = get_embedding(model, image, device)
        embeddings.append(embedding)

    embeddings = np.stack(embeddings)

    out_dir = PROJECT_ROOT / "data" / "case_db"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", embeddings)
    df.to_csv(out_dir / "case_metadata.csv", index=False)

    print(f"Save {len(df)} case embeddings, shape {embeddings.shape}")
    print(f"Output: {out_dir}/")

if __name__ == "__main__":
    main()