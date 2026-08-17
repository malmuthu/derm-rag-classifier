"""
PyTorch Dataset for HAM10000 lesion images.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMG_DIR_1 = PROJECT_ROOT / "data" / "raw" / "HAM10000_images_part_1"
IMG_DIR_2 = PROJECT_ROOT / "data" / "raw" / "HAM10000_images_part_2"

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class HAM10000Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = IMG_DIR_1 / f"{row['image_id']}.jpg"
        if not img_path.exists():
            img_path = IMG_DIR_2 / f"{row['image_id']}.jpg"

        image = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[row["dx"]]

        if self.transform:
            image = self.transform(image)

        return image, label

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def get_dataloaders(batch_size=32, num_workers=4, use_weighted_sampler=False, sampler_softening="sqrt"):
    processed_dir = PROJECT_ROOT / "data" / "processed"

    train_ds = HAM10000Dataset(processed_dir / "train.csv", transform=train_transform)
    val_ds = HAM10000Dataset(processed_dir / "val.csv", transform=eval_transform)
    test_ds = HAM10000Dataset(processed_dir / "test.csv", transform=eval_transform)

    if use_weighted_sampler:
        class_counts = train_ds.df["dx"].value_counts()

        if sampler_softening == "sqrt":
            class_weights = class_counts.map(lambda c: 1.0 / np.sqrt(c))
        else:
            class_weights = class_counts.map(lambda c: 1.0 / c)

        sample_weights = train_ds.df["dx"].map(class_weights).values
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader