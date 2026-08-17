"""
Search the FAISS case database for visually similar cases to a query image.
"""

import faiss
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_DB_DIR = PROJECT_ROOT / "data" / "case_db"

sys.path.append(str(PROJECT_ROOT / "src" / "data"))
from dataset import CLASSES

def load_case_db():
    index = faiss.read_index(str(CASE_DB_DIR / "faiss_index.bin"))
    metadata = pd.read_csv(CASE_DB_DIR / "case_metadata.csv")
    return index, metadata

def search_similar_cases(query_embedding, index, metdata, k=5):
    query = query_embedding.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query)

    similarities, indices = index.search(query, k)

    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        row = metdata.iloc[idx]
        results.append({
            "similarity": float(sim),
            "image_id": row["image_id"],
            "diagnosis": row["dx"],
            "dx_type": row["dx_type"],
            "age": row["age"],
            "sex": row["sex"],
            "localization": row["localization"],
        })
    return results