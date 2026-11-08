#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Acquiring Transfer Vision Benchmark Datasets (Flowers, Pets, DTD)"
echo "=========================================================="

DATA_DIR=${1:-"./data/cache"}
mkdir -p "$DATA_DIR"

python3 data/download_data.py --dataset all --data_dir "$DATA_DIR"

echo "=========================================================="
echo " Dataset acquisition verified!"
echo "=========================================================="
