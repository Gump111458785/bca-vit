#!/usr/bin/env bash
set -e

# ==============================================================================
# Full Reproduction Pipeline for BCA-ViT
# Fast evaluation profile: completes in ~1-2 minutes (Strictly under 15 minutes)
# Set EPOCHS=20 FULL_GRID=1 for extended paper reproduction run.
# ==============================================================================

EPOCHS=${1:-5}
echo "=========================================================="
echo " BCA-ViT Fast Reproduction Pipeline (Target Budget: < 15 min)"
echo " Configured Epochs per model: $EPOCHS"
echo "=========================================================="

echo ""
echo "=========================================================="
echo " 1. Checking CUDA Hardware Environment"
echo "=========================================================="
python3 -c "import torch; assert torch.cuda.is_available(), 'CRITICAL: CUDA GPU is required!'"
echo "CUDA Hardware detected: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

echo ""
echo "=========================================================="
echo " 2. Downloading & Verifying Pretrained Weights"
echo "=========================================================="
bash scripts/download_weights.sh "./weights"

echo ""
echo "=========================================================="
echo " 3. Downloading Transfer Benchmark Datasets"
echo "=========================================================="
bash scripts/download_data.sh "./data/cache"

echo ""
echo "=========================================================="
echo " 4. Benchmarking Parameter Efficiency & FLOPs (Table 2)"
echo "=========================================================="
python3 scripts/benchmark_efficiency.py

echo ""
echo "=========================================================="
echo " 5. Running Main Transfer Learning Experiments (Table 1)"
echo "=========================================================="
if [ "$FULL_GRID" = "1" ]; then
    DATASETS=("flowers102" "pets" "dtd")
    MODELS=("linear_probe" "full_ft" "lora" "vpt" "bca_vit")
    for d in "${DATASETS[@]}"; do
        for m in "${MODELS[@]}"; do
            echo "--> Training: Model=$m on Dataset=$d ($EPOCHS epochs)"
            python3 scripts/train.py --model "$m" --dataset "$d" --epochs "$EPOCHS" --batch_size 32 --output_dir "./checkpoints"
        done
    done
else
    echo "--> Running core model: BCA-ViT on Flowers-102 ($EPOCHS epochs)"
    python3 scripts/train.py --model bca_vit --dataset flowers102 --epochs "$EPOCHS" --batch_size 32 --output_dir "./checkpoints"

    echo "--> Running baseline model: Linear Probe on Flowers-102 ($EPOCHS epochs)"
    python3 scripts/train.py --model linear_probe --dataset flowers102 --epochs "$EPOCHS" --batch_size 32 --output_dir "./checkpoints"
fi

echo ""
echo "=========================================================="
echo " 6. Running Ablation Study on Flowers-102 (Table 3)"
echo "=========================================================="
echo "--> Ablation: BCA-ViT without BCA adapter ($EPOCHS epochs)"
python3 scripts/train.py --model bca_vit --dataset flowers102 --no_bca --epochs "$EPOCHS" --output_dir "./checkpoints/ablation_no_bca"

echo ""
echo "=========================================================="
echo " 7. Verifying Reproduction Metrics"
echo "=========================================================="
python3 scripts/verify_reproduction.py

echo "=========================================================="
echo " Transfer reproduction suite completed (Time < 15 min)!"
echo "=========================================================="
