"""Transfer vision dataset loaders with 224x224 resizing and ImageNet normalization.
"""

import os
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from .download_data import verify_dataset, download_dataset, SUPPORTED_DATASETS

DATASET_CONFIGS = {
    "flowers102": {"num_classes": 102, "img_size": 224},
    "pets": {"num_classes": 37, "img_size": 224},
    "dtd": {"num_classes": 47, "img_size": 224},
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Must be in {SUPPORTED_DATASETS}")
    return DATASET_CONFIGS[dataset_name]


def get_dataloaders(
    dataset_name: str,
    data_dir: str = "./data/cache",
    batch_size: int = 32,
    num_workers: int = 2,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    dataset_name = dataset_name.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    target_dir = os.path.join(data_dir, dataset_name)
    if not verify_dataset(dataset_name, data_dir):
        if download:
            try:
                download_dataset(dataset_name, data_dir)
            except Exception as e:
                print(f"[Dataset Download Warning] Automatic download failed: {e}")

    cfg = get_dataset_info(dataset_name)
    num_classes = cfg["num_classes"]

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    try:
        if dataset_name == "flowers102":
            train_ds = datasets.Flowers102(root=target_dir, split="train", transform=train_tf, download=download)
            test_ds = datasets.Flowers102(root=target_dir, split="test", transform=test_tf, download=download)
        elif dataset_name == "pets":
            train_ds = datasets.OxfordIIITPet(root=target_dir, split="trainval", transform=train_tf, download=download)
            test_ds = datasets.OxfordIIITPet(root=target_dir, split="test", transform=test_tf, download=download)
        elif dataset_name == "dtd":
            train_ds = datasets.DTD(root=target_dir, split="train", transform=train_tf, download=download)
            test_ds = datasets.DTD(root=target_dir, split="test", transform=test_tf, download=download)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load authentic benchmark dataset '{dataset_name}' from '{target_dir}' ({e}).\n"
            f"Please execute 'bash scripts/download_data.sh' first to acquire the official benchmark datasets."
        ) from e

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, test_loader
