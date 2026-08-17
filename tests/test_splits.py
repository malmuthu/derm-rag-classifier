"""
Tests for the lesion-level train/val/test split.
Run: pytest tests/test_splits.py -v
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_splits():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train, val, test

def test_no_leakage_between_splits():
    train, val, test = load_splits()

    train_ids = set(train["lesion_id"])
    val_ids = set(val["lesion_id"])
    test_ids = set(test["lesion_id"])

    assert train_ids.isdisjoint(val_ids), "Lesion overlap between train and val"
    assert train_ids.isdisjoint(test_ids), "Lesion overlap between train and test"
    assert val_ids.isdisjoint(test_ids), "Lesion overlap between val and test"

def test_all_classes_present_in_every_split():
    train, val, test = load_splits()
    all_classes = set(train["dx"].unique())

    for name, split_df in [("val", val), ("test", test)]:
        missing = all_classes - set(split_df["dx"].unique())
        assert not missing, f"{name} split is missing classes: {missing}"

def test_no_missing_images_lost():
    train, val, test = load_splits()
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "HAM10000_metadata.csv")

    total_split_rows = len(train) + len(val) + len(test)
    assert total_split_rows == len(raw), (
        f"Row count mismatch: splits have {total_split_rows}, "
        f"raw has {len(raw)}"
    )

def test_class_proportions_roughly_preserved():
    train, val, test = load_splits()
    raw_props = train["dx"].value_counts(normalize=True)

    for name, split_df in [("val", val), ("test", test)]:
        split_props = split_df["dx"].value_counts(normalize=True)
        for cls in raw_props.index:
            train_pct = raw_props.get(cls, 0)
            split_pct = split_props.get(cls, 0)

            assert abs(train_pct - split_pct) < 0.05, (
                f"{name}/{cls}: train={train_pct:.3f} vs {name}={split_pct:.3f}"
            )