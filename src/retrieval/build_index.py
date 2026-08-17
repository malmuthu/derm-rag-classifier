"""
Build a FAISS similarity search index from precomputed case embeddings.
Run: python src/retrieval/build_index.py
"""

import faiss
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_DB_DIR = PROJECT_ROOT / "data" / "case_db"

def main():
    embeddings = np.load(CASE_DB_DIR / "embeddings.npy").astype("float32")

    # Normalize each embedding to unit length so inner product is similar to cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(CASE_DB_DIR / "faiss_index.bin"))
    print(f"Built FAISS index with {index.ntotal} vectors, dim={dim}")

if __name__ == "__main__":
    main()