import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from utils.config import Config


def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Transforms ────────────────────────────────────────────────
def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE + 20, Config.IMG_SIZE + 20)),
        transforms.RandomCrop(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(Config.MEAN, Config.STD),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(Config.MEAN, Config.STD),
    ])

    return train_transforms, val_transforms


# ── Dataset class ─────────────────────────────────────────────
class RetinopathyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["id_code"] + ".png")
        image    = Image.open(img_path).convert("RGB")
        label    = int(row["diagnosis"])
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Sampler for class imbalance ───────────────────────────────
def get_weighted_sampler(train_df):
    class_counts  = Counter(train_df["diagnosis"].tolist())
    class_weights = {c: 1.0 / class_counts[c] for c in class_counts}
    sample_weights = [
        class_weights[int(train_df.iloc[i]["diagnosis"])]
        for i in range(len(train_df))
    ]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True
    )


# ── Build splits + DataLoaders ────────────────────────────────
def get_dataloaders(df):
    # Optional per-class cap
    if Config.MAX_PER_CLASS is not None:
        df = (
            df.groupby("diagnosis", group_keys=False)
              .apply(lambda g: g.sample(
                  min(len(g), Config.MAX_PER_CLASS),
                  random_state=Config.SEED))
              .reset_index(drop=True)
        )

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["diagnosis"],
        random_state=Config.SEED
    )

    train_tf, val_tf = get_transforms()

    train_dataset = RetinopathyDataset(train_df, Config.TRAIN_IMGS, train_tf)
    val_dataset   = RetinopathyDataset(val_df,   Config.TRAIN_IMGS, val_tf)

    sampler = get_weighted_sampler(train_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, train_df, val_df, train_dataset, val_dataset
