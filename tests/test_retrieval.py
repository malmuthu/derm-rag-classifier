"""
Tests for the retrieval layer: case database integrity and retrieval quality.
Run: pytest tests/test_retrieval.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import torch
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src" / "data"))
sys.path.append(str(PROJECT_ROOT / "src" / "models"))
sys.path.append(str(PROJECT_ROOT / "src" / "retrieval"))
sys.path.append(str(PROJECT_ROOT / "src" / "utils"))

from config import load_config
from embeddings import load_embedding_model, get_embedding
from search import load_case_db, search_similar_cases

CASE_DB_DIR = PROJECT_ROOT / "data" / "case_db"
IMG_DIR_1 = PROJECT_ROOT / "data" / "raw" / "HAM10000_images_part_1"
IMG_DIR_2 = PROJECT_ROOT / "data" / "raw" / "HAM10000_images_part_2"

@pytest.fixture(scope="module")
def embedding_model():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    retrieval_config = load_config(PROJECT_ROOT / "configs" / "retrieval.yaml")
    checkpoint_path = PROJECT_ROOT / retrieval_config["embedding_checkpoint"]
    model = load_embedding_model(checkpoint_path, device)
    return model, device

@pytest.fixture(scope="module")
def case_db():
    index, metadata = load_case_db()
    return index, metadata

def load_image(image_id):
    path = IMG_DIR_1 / f"{image_id}.jpg"
    if not path.exists():
        path = IMG_DIR_2 / f"{image_id}.jpg"
    return Image.open(path).convert("RGB")

def test_case_db_row_alignment():
    embeddings = np.load(CASE_DB_DIR / "embeddings.npy")
    metadata = pd.read_csv(CASE_DB_DIR / "case_metadata.csv")

    assert embeddings.shape[0] == len(metadata), (
        "Embedding count doesn't match metadata row count - "
        "row alignment between embeddings.npy and case_metadata.csv is broken."
    )

def test_faiss_index_matches_case_db(case_db):
    index, metadata = case_db
    assert index.ntotal == len(metadata), (
        "FAISS index vector count doesn't match case metadata row count."
    )

def test_query_retrieves_expected_k(embedding_model, case_db):
    model, device = embedding_model
    index, metadata = case_db

    image = load_image(metadata.iloc[0]["image_id"])
    embedding = get_embedding(model, image, device)
    results = search_similar_cases(embedding, index, metadata, k=5)

    assert len(results) == 5

def test_query_against_own_case_db_retrieves_itself(embedding_model, case_db):
    """
    Embedding an image that is in the case database and search should return that
    same image as the top-1, confirming embedding + search pipeline is consistent.
    """
    model, device = embedding_model
    index, metadata = case_db

    query_row = metadata.iloc[0]
    image = load_image(query_row["image_id"])
    embedding = get_embedding(model, image, device)
    results = search_similar_cases(embedding, index, metadata, k=1)

    assert results[0]["image_id"] == query_row["image_id"], (
        "Querying an image already in the case DB did not retrieve "
        "itself as the top-1 match - embedding/search may be inconsistent."
    )
    assert results[0]["similarity"] > 0.99, (
        f"Expected near-1.0 self-similarity, got {results[0]['similarity']:.4f}"
    )

def test_diagnosis_match_rate_across_classes(embedding_model, case_db):
    """
    For one held-out test set query per class, check that the majority of the 
    top-5 retrieved classes share the query's true diagnosis.
    """
    model, device = embedding_model
    index, metadata = case_db

    test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
    sample_rows = test_df.groupby("dx").head(1)

    classes_with_good_match = 0
    for _, row in sample_rows.iterrows():
        image = load_image(row["image_id"])
        embedding = get_embedding(model, image, device)
        results = search_similar_cases(embedding, index, metadata, k=5)

        matches = sum(1 for r in results if r["diagnosis"] == row["dx"])
        if matches >=3:
            classes_with_good_match += 1

    total_classes = len(sample_rows)
    match_rate = classes_with_good_match / total_classes

    assert match_rate >= 0.7, (
        f"Only {classes_with_good_match}/{total_classes} classes had "
        f"good retrieval match rate (>=3/5) - retrieval quality may have regressed."
    )