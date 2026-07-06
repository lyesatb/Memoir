# src/data.py
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2,0.2,0.2,0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    return transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

class ImageMultiTaskDataset(Dataset):
    def __init__(self, df, img_root, target_cols, transforms):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transforms = transforms
        self.target_cols = target_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / row["filename"]
        img = Image.open(img_path).convert("RGB")
        x = self.transforms(img)
        y = {}
        for c in self.target_cols:
            y[c] = row[c]
        return x, y, str(img_path)

def make_dataloaders(train_df, val_df, test_df, img_root, target_cols,
                     img_size=224, batch_size=16, num_workers=4):
    train_ds = ImageMultiTaskDataset(train_df, img_root, target_cols, get_transforms(img_size, train=True))
    val_ds = ImageMultiTaskDataset(val_df, img_root, target_cols, get_transforms(img_size, train=False))
    test_ds = ImageMultiTaskDataset(test_df, img_root, target_cols, get_transforms(img_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
