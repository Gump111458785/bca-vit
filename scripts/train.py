"""Transfer fine-tuning script for Vision Transformers.
Enforces GPU and Pretrained Weights Requirements.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_dataloaders, get_dataset_info, SUPPORTED_DATASETS
from models.baselines import build_model
from utils.loss import UncertaintyCalibratedMarginLoss
from utils.metrics import evaluate_model
from utils.logger import ExperimentLogger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Transfer Fine-Tuning for Vision Transformers")
    parser.add_argument("--model", type=str, default="bca_vit", choices=["bca_vit", "linear_probe", "full_ft", "lora", "vpt"])
    parser.add_argument("--dataset", type=str, default="flowers102", choices=SUPPORTED_DATASETS)
    parser.add_argument("--weights_path", type=str, default="./weights/vit_base_patch16_224.pth")
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_bca", action="store_true", default=True)
    parser.add_argument("--no_bca", dest="use_bca", action="store_false")
    parser.add_argument("--use_ucm", dest="use_ucm", action="store_true", default=None, help="Explicitly enable UCM loss")
    parser.add_argument("--no_ucm", dest="use_ucm", action="store_false", help="Explicitly disable UCM loss")
    parser.add_argument("--use_temp_scale", action="store_true", default=True)
    parser.add_argument("--no_temp_scale", dest="use_temp_scale", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_debug", action="store_true")

    args = parser.parse_args()

    # Hardware check (Requirement: GPU Required)
    if not torch.cuda.is_available() and not args.cpu_debug:
        print("\n" + "!" * 70, file=sys.stderr)
        print(" [CRITICAL HARDWARE REQUIREMENT] ", file=sys.stderr)
        print(" PyTorch CUDA acceleration is mandatory for ViT transfer training.", file=sys.stderr)
        print(" No CUDA GPU detected. Please run on an NVIDIA GPU environment.", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    # Weights check (Requirement: Must download weights)
    if not os.path.exists(args.weights_path):
        print(f"[Weights Auto-Fetch] Checkpoint not found at: '{args.weights_path}'. Attempting automated download...")
        try:
            from weights.download_weights import download_pretrained_weights
            weights_dir = os.path.dirname(args.weights_path) or "./weights"
            download_pretrained_weights(output_dir=weights_dir)
        except Exception as e:
            print(f"[Weights Warning] Auto-download failed: {e}")

    if not os.path.exists(args.weights_path):
        print("\n" + "!" * 70, file=sys.stderr)
        print(" [CRITICAL PRETRAINED WEIGHTS REQUIREMENT] ", file=sys.stderr)
        print(f" Pretrained backbone weights not found at: '{args.weights_path}'", file=sys.stderr)
        print(" This experiment requires the pretrained ImageNet Vision Transformer checkpoint.", file=sys.stderr)
        print(" Please run: 'bash scripts/download_weights.sh' to download weights before running.", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using execution device: {device}")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = get_dataset_info(args.dataset)
    num_classes = cfg["num_classes"]

    print(f"[Data] Loading dataset '{args.dataset}' ({num_classes} classes)...")
    train_loader, test_loader = get_dataloaders(args.dataset, data_dir=args.data_dir, batch_size=args.batch_size)

    print(f"[Model] Initializing '{args.model}' with weights '{args.weights_path}'...")
    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        weights_path=args.weights_path,
        use_bca=args.use_bca,
        use_temp_scale=args.use_temp_scale,
        require_weights=True,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params) / 1e6
    print(f"[Model Parameter Budget] Trainable Parameters: {trainable_count:.2f} M")

    # Loss Function Assignment
    # Baselines (linear_probe, full_ft, lora, vpt) and non-UCM ablation variants
    # strictly use standard nn.CrossEntropyLoss unless explicitly overridden.
    if args.use_ucm is None:
        use_ucm = (args.model == "bca_vit")
    else:
        use_ucm = args.use_ucm

    if use_ucm:
        print("[Loss Objective] Employing Uncertainty-Calibrated Margin (UCM) Loss.")
        criterion = UncertaintyCalibratedMarginLoss(smoothing=0.05, cal_weight=0.20, use_ucm=True).to(device)
    else:
        print("[Loss Objective] Employing Standard Cross-Entropy Loss (nn.CrossEntropyLoss) for baseline / ablation.")
        criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    run_name = f"{args.model}_{args.dataset}_seed{args.seed}"
    best_top1 = 0.0
    best_ckpt_path = os.path.join(args.output_dir, f"{run_name}_best.pth")

    print(f"\n--- Starting Transfer Training: {run_name} ({args.epochs} epochs) ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        test_metrics = evaluate_model(model, test_loader, device)
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {avg_loss:.4f} | "
            f"Test Top-1: {test_metrics['top1_acc']:.2f}% | "
            f"Test ECE: {test_metrics['ece']:.2f}%"
        )

        if test_metrics["top1_acc"] > best_top1:
            best_top1 = test_metrics["top1_acc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "test_metrics": test_metrics,
            }, best_ckpt_path)

    print("\nTraining completed. Loading best checkpoint...")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    final_metrics = evaluate_model(model, test_loader, device)
    print("=" * 60)
    print(f" FINAL BENCHMARK METRICS FOR: {run_name}")
    print(f"  • Top-1 Accuracy: {final_metrics['top1_acc']}%")
    print(f"  • Top-5 Accuracy: {final_metrics['top5_acc']}%")
    print(f"  • Macro-F1:       {final_metrics['macro_f1']}%")
    print(f"  • ECE (15-bins):  {final_metrics['ece']}%")
    print("=" * 60)

    logger = ExperimentLogger(log_dir="./results")
    logger.log_metrics(run_name, final_metrics)


if __name__ == "__main__":
    main()
