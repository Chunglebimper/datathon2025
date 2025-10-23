"""
data.py
--------
Defines MosquitoDataset and DataLoader builders.
- Loads (image_path, label) pairs from JSON splits.
- Applies augmentations through the transform pipeline.
- Supports weighted sampling to handle class imbalance.
"""
from pathlib import Path
import json
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageOps
from sklearn.preprocessing import LabelEncoder
import numpy as np

class MosquitoDataset(Dataset):
    def __init__(self, pairs, transform=None, label_encoder=None):
        self.items = pairs
        self.transform = transform
        self.le = label_encoder or self._fit_label_encoder()

    def _fit_label_encoder(self):
        labels = [lab for _, lab in self.items]
        le = LabelEncoder(); le.fit(labels)
        return le

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        if self.transform: img = self.transform(img)
        y = self.le.transform([label])[0]
        return img, y

    @property
    def num_classes(self): return len(self.le.classes_)
    @property
    def classes(self): return list(self.le.classes_)


def build_loaders(splits_file: str, img_size: int, batch_size: int, num_workers: int, balance_sampler: bool, transforms):
    with open(splits_file, "r") as f:
        splits = json.load(f)

    train_pairs = splits["train"]
    val_pairs   = splits["val"]

    train_ds = MosquitoDataset(train_pairs, transform=transforms.train)
    le = train_ds.le
    val_ds = MosquitoDataset(val_pairs, transform=transforms.eval, label_encoder=le)

    if balance_sampler:
        labels = np.array([le.transform([lab])[0] for _, lab in train_pairs])
        counts = np.bincount(labels, minlength=train_ds.num_classes).astype(float)
        weights = 1.0 / np.maximum(counts, 1)
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    meta = {"num_classes": train_ds.num_classes, "classes": train_ds.classes}
    return train_loader, val_loader, meta
