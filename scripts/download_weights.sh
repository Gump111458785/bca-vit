#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Acquiring Pretrained Vision Transformer (ViT-B/16) Weights"
echo " Destination: ./weights/vit_base_patch16_224.pth"
echo "=========================================================="

WEIGHTS_DIR=${1:-"./weights"}
mkdir -p "$WEIGHTS_DIR"

python3 weights/download_weights.py --model vit_base_patch16_224 --output_dir "$WEIGHTS_DIR"

echo "=========================================================="
echo " Pretrained backbone weights successfully verified!"
echo "=========================================================="
