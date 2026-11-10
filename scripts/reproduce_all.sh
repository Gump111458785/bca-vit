#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " 1. Checking CUDA Hardware & Pretrained Foundation Weights"
echo "=========================================================="
python3 -c "import torch; assert torch.cuda.is_available(), 'CRITICAL: CUDA GPU is required!'"
echo "CUDA Hardware detected: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

echo ""
echo "=========================================================="
echo " 2. Downloading & Verifying Pretrained Weights (ViT-B/16)"
echo "=========================================================="
bash scripts/download_weights.sh "./weights"

echo ""
echo "=========================================================="
echo " 3. Downloading Transfer Benchmark Datasets"
echo "=========================================================="
bash scripts/download_data.sh "./data/cache"

echo ""
echo "=========================================================="
echo " 4. Profiling Parameter-Budget & GPU Efficiency (Table 2)"
echo "=========================================================="
python3 scripts/benchmark_efficiency.py

echo ""
echo "=========================================================="
echo " 5. Running Main Transfer Learning Experiments (Table 1)"
echo "=========================================================="
DATASETS=("flowers102" "pets" "dtd")
MODELS=("linear_probe" "full_ft" "lora" "vpt" "bca_vit")

for d in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        echo "--> Training & Evaluating: Model=$m on Dataset=$d"
        python3 scripts/train.py \
            --model "$m" \
            --dataset "$d" \
            --epochs 20 \
            --batch_size 32 \
            --output_dir "./checkpoints"
    done
done

echo ""
echo "=========================================================="
echo " 6. Running Ablation Study on Flowers-102 (Table 3)"
echo "=========================================================="
# (a) Full Fine-Tuning Baseline
python3 scripts/train.py --model full_ft --dataset flowers102 --no_bca --no_ucm --no_temp_scale --output_dir "./checkpoints/ablation_a"
python3 scripts/train.py --model bca_vit --dataset flowers102 --use_bca --no_ucm --no_temp_scale --output_dir "./checkpoints/ablation_b"
python3 scripts/train.py --model bca_vit --dataset flowers102 --no_bca --use_ucm --no_temp_scale --output_dir "./checkpoints/ablation_c"
python3 scripts/train.py --model bca_vit --dataset flowers102 --use_bca --use_ucm --no_temp_scale --output_dir "./checkpoints/ablation_d"

echo ""
echo "=========================================================="
echo " 7. Verifying Reproduction Metrics"
echo "=========================================================="
python3 scripts/verify_reproduction.py

echo "=========================================================="
echo " Transfer reproduction suite completed!"
echo "=========================================================="
