from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


FER_RAW_SIZE = 48  # FER2013 images are always 48x48 in the CSV


def csv_to_imagefolder(cfg, fer_csv_path: str):
    """
    Converts fer2013.csv -> ImageFolder structure:
      processed/
        train/<label>/*.png
        test/<label>/*.png

    IMPORTANT:
    - FER2013 pixels are 48x48 in the CSV, so this conversion MUST use 48,
      NOT cfg.image_size (because cfg.image_size might be 224 for training).
    """
    out_root = Path(cfg.processed_dir)
    train_root = out_root / "train"
    test_root = out_root / "test"

    # If already processed, skip
    if train_root.exists() and test_root.exists() and any(train_root.rglob("*.png")):
        return str(train_root), str(test_root)

    df = pd.read_csv(fer_csv_path)
    if "emotion" not in df.columns or "pixels" not in df.columns:
        raise ValueError(f"CSV must contain emotion,pixels. Found: {list(df.columns)}")

    # Some mirrors may not have Usage column
    if "Usage" not in df.columns:
        df["Usage"] = "Training"
        idx = np.arange(len(df))
        np.random.shuffle(idx)
        test_n = int(0.1 * len(df))
        df.loc[idx[:test_n], "Usage"] = "Test"

    # Create label folders
    for split_root in [train_root, test_root]:
        for label in range(7):
            (split_root / str(label)).mkdir(parents=True, exist_ok=True)

    def save_row(row, split_root: Path, i: int):
        label = int(row["emotion"])
        arr = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        if arr.size != FER_RAW_SIZE * FER_RAW_SIZE:
            return False
        img = arr.reshape(FER_RAW_SIZE, FER_RAW_SIZE)
        Image.fromarray(img, mode="L").save(split_root / str(label) / f"{i:06d}.png")
        return True

    ok = 0
    for i, row in df.iterrows():
        usage = str(row["Usage"]).lower()
        if "train" in usage:
            ok += int(save_row(row, train_root, i))
        else:
            ok += int(save_row(row, test_root, i))

    return str(train_root), str(test_root)


def make_transforms(cfg):
    # choose normalization based on number of channels
    if cfg.input_channels == 3:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5,)
        std = (0.5,)

    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=cfg.input_channels),
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08))
        ], p=0.9),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=cfg.input_channels),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # DEBUG guard (optional)
    names = []
    for t in train_tfms.transforms:
        names.append(type(t).__name__)
        if hasattr(t, "transforms"):
            names.extend([type(x).__name__ for x in t.transforms])
    if "RandomErasing" in names:
        raise RuntimeError(f"RandomErasing is still present in train transforms: {names}")

    return train_tfms, test_tfms


def compute_class_counts(imagefolder_dataset: ImageFolder, indices=None, num_classes=7):
    if indices is None:
        labels = imagefolder_dataset.targets
    else:
        labels = [imagefolder_dataset.targets[i] for i in indices]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    return counts


def make_loaders(cfg, train_root: str, test_root: str):
    train_tfms, test_tfms = make_transforms(cfg)

    full_train = ImageFolder(root=train_root, transform=train_tfms)
    test_ds = ImageFolder(root=test_root, transform=test_tfms)

    # Train/Val split
    val_len = int(len(full_train) * cfg.val_split)
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(
        full_train,
        lengths=[train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    sampler = None
    shuffle = True

    if cfg.use_weighted_sampler:
        counts = compute_class_counts(full_train, indices=train_ds.indices, num_classes=7)
        class_weights = 1.0 / counts
        labels = [full_train.targets[i] for i in train_ds.indices]
        sample_weights = [class_weights[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "mps"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "mps"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "mps"),
    )

    return full_train, train_ds, train_loader, val_loader, test_loader