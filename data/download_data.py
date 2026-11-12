"""Transfer vision benchmark dataset downloader and verifier.
Supports Flowers-102, Oxford-IIIT Pets, and DTD (Describable Textures).
"""

import os
import sys
import argparse
import torch
from torchvision import datasets, transforms

SUPPORTED_DATASETS = ["flowers102", "pets", "dtd"]


def download_dataset(dataset_name: str, root_dir: str = "./data/cache") -> bool:
    dataset_name = dataset_name.lower()
    os.makedirs(root_dir, exist_ok=True)
    target_dir = os.path.join(root_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)

    print(f"[Dataset] Downloading & verifying '{dataset_name}' into '{target_dir}'...")
    try:
        if dataset_name == "flowers102":
            datasets.Flowers102(root=target_dir, split="train", download=True)
            datasets.Flowers102(root=target_dir, split="test", download=True)
        elif dataset_name in ["pets", "oxford_pets"]:
            datasets.OxfordIIITPet(root=target_dir, split="trainval", download=True)
            datasets.OxfordIIITPet(root=target_dir, split="test", download=True)
        elif dataset_name == "dtd":
            datasets.DTD(root=target_dir, split="train", download=True)
            datasets.DTD(root=target_dir, split="test", download=True)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        print(f"[Dataset] Successfully prepared '{dataset_name}'.")
        return True
    except Exception as e:
        raise RuntimeError(
            f"Failed to acquire official '{dataset_name}' dataset ({e}).\n"
            f"Please check your internet connection or download '{dataset_name}' manually."
        ) from e


def verify_dataset(dataset_name: str, root_dir: str = "./data/cache") -> bool:
    target_dir = os.path.join(root_dir, dataset_name.lower())
    if not os.path.isdir(target_dir):
        return False
    return len(os.listdir(target_dir)) > 0


def download_all_datasets(root_dir: str = "./data/cache"):
    print("=" * 60)
    print("Acquiring Transfer Benchmark Datasets (Flowers102, Pets, DTD)")
    print("=" * 60)
    for d in SUPPORTED_DATASETS:
        download_dataset(d, root_dir)
    print("\nAll transfer benchmark datasets are ready.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download transfer datasets")
    parser.add_argument("--dataset", type=str, default="all", choices=["all"] + SUPPORTED_DATASETS)
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    args = parser.parse_args()

    if args.dataset == "all":
        download_all_datasets(args.data_dir)
    else:
        download_dataset(args.dataset, args.data_dir)
