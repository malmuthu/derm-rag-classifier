"""
Create a lesion-level, class-stratified train/val/test split for HAM10000.
Run once: python src/data/make_splits.py
Outputs:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_META = PROJECT_ROOT / "data" / "raw" / "HAM10000_metadata.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
SEED = 42

def make_splits():
    df = pd.read_csv(RAW_META)

    # One row per unique lesion
    lesion_df = df.drop_duplicates("lesion_id")[["lesion_id", "dx"]]

    train_lesions, other_lesions = train_test_split(
        lesion_df,
        test_size=0.30,
        stratify=lesion_df["dx"],
        random_state=SEED,
    )

    val_lesions, test_lesions = train_test_split(
        other_lesions,
        test_size=0.50,
        stratify=other_lesions["dx"],
        random_state=SEED,
    )

    return train_lesions, val_lesions, test_lesions

def assign_splits_to_images(df, train_lesions, val_lesions, test_lesions):
    train_ids = set(train_lesions["lesion_id"])
    val_ids = set(val_lesions["lesion_id"])
    test_ids = set(test_lesions["lesion_id"])

    train_df = df[df["lesion_id"].isin(train_ids)].copy()
    val_df = df[df["lesion_id"].isin(val_ids)].copy()
    test_df = df[df["lesion_id"].isin(test_ids)].copy()

    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

def verify_no_leakage(train_df, val_df, test_df):
    train_ids = set(train_df["lesion_id"])
    val_ids = set(val_df["lesion_id"])
    test_ids = set(test_df["lesion_id"])

    assert train_ids.isdisjoint(val_ids), "Leakage: lesion in both train and val"
    assert train_ids.isdisjoint(test_ids), "Leakage: lesion in both train and test"
    assert val_ids.isdisjoint(test_ids), "Leakage: lesion in both val and test"
    print("No lesion-level leakage between splits.")

def report_split_sizes(train_df, val_df, test_df):
    total = len(train_df) + len(val_df) + len(test_df)
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name:5s}: {len(d):5d} images "
              f"({len(d)/total*100:5.1f}%), "
              f"{d['lesion_id'].nunique():5d} unique lesions")

    print("\nClass distribution per split (%):")
    dist = pd.concat({
        "train": train_df["dx"].value_counts(normalize=True) * 100,
        "val": val_df["dx"].value_counts(normalize=True) * 100,
        "test": test_df["dx"].value_counts(normalize=True) * 100,
    }, axis=1).round(1)
    print(dist)

if __name__ == "__main__":
    train_lesions, val_lesions, test_lesions = make_splits()
    df = pd.read_csv(RAW_META)
    train_df, val_df, test_df = assign_splits_to_images(df, train_lesions, val_lesions, test_lesions)
    verify_no_leakage(train_df, val_df, test_df)
    report_split_sizes(train_df, val_df, test_df)
    save_splits(train_df, val_df, test_df)
    print(f"\nSplits saved to {OUT_DIR}/")