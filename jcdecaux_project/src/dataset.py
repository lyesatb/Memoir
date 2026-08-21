import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
import numpy as np

from config.config import (
    IMG_SIZE, BATCH_SIZE, DEVICE, NUM_WORKERS,
    SPLIT, USE_WEIGHTED_SAMPLER, SAMPLER_TARGET
)
from src.img_utils import safe_open_rgb

class MultiTaskDataset(Dataset):
    def __init__(self, df, target_cols, col_types, transforms=None):
        self.df = df.reset_index(drop=True)
        self.target_cols = target_cols
        self.col_types = col_types
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        img = safe_open_rgb(img_path)

        if self.transforms:
            img = self.transforms(img)

        targets = {}
        for c in self.target_cols:
            val = row[c + "_enc"]
            ttype = self.col_types[c]
            if ttype == "categorical":
                targets[c] = torch.tensor(int(val), dtype=torch.long)
            elif ttype == "binary":
                targets[c] = torch.tensor(float(val), dtype=torch.float32)
            else:
                targets[c] = torch.tensor(float(val), dtype=torch.float32)

        return img, targets, img_path


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])


def build_dataloaders(df, target_cols, col_types):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train = int(SPLIT[0]*n)
    n_val = int(SPLIT[1]*n)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]

    train_ds = MultiTaskDataset(train_df, target_cols, col_types, get_transforms(True))
    val_ds = MultiTaskDataset(val_df, target_cols, col_types, get_transforms(False))
    test_ds = MultiTaskDataset(test_df, target_cols, col_types, get_transforms(False))

    sampler = None
    if USE_WEIGHTED_SAMPLER and SAMPLER_TARGET:
        labels = train_df[SAMPLER_TARGET+"_enc"].values
        class_count = np.bincount(labels)
        weights = 1. / class_count
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        targets = {c: torch.stack([b[1][c] for b in batch]) for c in target_cols}
        paths = [b[2] for b in batch]
        return imgs, targets, paths

    return (
        DataLoader(train_ds, BATCH_SIZE, shuffle=sampler is None, sampler=sampler,
                   num_workers=NUM_WORKERS, collate_fn=collate),
        DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate),
        DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate)
    )
