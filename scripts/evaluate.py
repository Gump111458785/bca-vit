"""Transfer model evaluation script.
Requires GPU and Pretrained Weights.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_dataloaders, get_dataset_info, SUPPORTED_DATASETS
from models.baselines import build_model
from utils.metrics import evaluate_model
from utils.logger import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT transfer models")
    parser.add_argument("--model", type=str, default="bca_vit", choices=["bca_vit", "linear_probe", "full_ft", "lora", "vpt"])
    parser.add_argument("--dataset", type=str, default="flowers102", choices=SUPPORTED_DATASETS)
    parser.add_argument("--weights_path", type=str, default="./weights/vit_base_patch16_224.pth")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_bca", action="store_true", default=True)
    parser.add_argument("--no_bca", dest="use_bca", action="store_false")
    parser.add_argument("--use_temp_scale", action="store_true", default=True)
    parser.add_argument("--no_temp_scale", dest="use_temp_scale", action="store_false")
    parser.add_argument("--cpu_debug", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu_debug:
        print("\n[CRITICAL] CUDA GPU is required for transfer evaluation.\n", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.weights_path):
        print(f"\n[CRITICAL] Pretrained weights not found at: '{args.weights_path}'", file=sys.stderr)
        print("Please run: 'bash scripts/download_weights.sh' to download weights first.\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_dataset_info(args.dataset)

    _, test_loader = get_dataloaders(args.dataset, data_dir=args.data_dir, batch_size=args.batch_size)
    model = build_model(
        model_name=args.model,
        num_classes=cfg["num_classes"],
        weights_path=args.weights_path,
        use_bca=args.use_bca,
        use_temp_scale=args.use_temp_scale,
        require_weights=True,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    model = model.to(device)
    metrics = evaluate_model(model, test_loader, device)

    print("\n" + "=" * 55)
    print(f" BENCHMARK EVALUATION: {args.model.upper()} on {args.dataset.upper()}")
    print("=" * 55)
    print(f"  • Top-1 Accuracy: {metrics['top1_acc']:.2f} %")
    print(f"  • Top-5 Accuracy: {metrics['top5_acc']:.2f} %")
    print(f"  • Macro-F1:       {metrics['macro_f1']:.2f} %")
    print(f"  • ECE (15-bins):  {metrics['ece']:.2f} %")
    print(f"  • Total Samples:  {metrics['total_samples']}")
    print("=" * 55 + "\n")

    logger = ExperimentLogger()
    run_name = f"eval_{args.model}_{args.dataset}"
    logger.log_metrics(run_name, metrics)


if __name__ == "__main__":
    main()
