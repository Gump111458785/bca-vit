"""Baseline transfer learning methods for Vision Transformers.
Includes Linear Probe, Full Fine-Tuning, LoRA, and VPT.
"""

import torch
import torch.nn as nn
from .vit_backbone import VisionTransformerBackbone
from .bca_vit import BCAViT


class LinearProbeViT(nn.Module):
    def __init__(self, num_classes=102, weights_path="./weights/vit_base_patch16_224.pth", require_weights=True):
        super().__init__()
        self.backbone = VisionTransformerBackbone(weights_path=weights_path, require_weights=require_weights)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


class FullFineTuneViT(nn.Module):
    def __init__(self, num_classes=102, weights_path="./weights/vit_base_patch16_224.pth", require_weights=True):
        super().__init__()
        self.backbone = VisionTransformerBackbone(weights_path=weights_path, require_weights=require_weights)
        # Backbone is fully trainable
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8):
        super().__init__()
        self.r = r
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = 1.0

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRAViT(nn.Module):
    def __init__(self, num_classes=102, weights_path="./weights/vit_base_patch16_224.pth", require_weights=True):
        super().__init__()
        self.backbone = VisionTransformerBackbone(weights_path=weights_path, require_weights=require_weights)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.loras = nn.ModuleList([
            LoRALinear(768, 768, r=8) for _ in range(12)
        ])
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        b = x.size(0)
        x = self.backbone.patch_embed(x)
        cls_tokens = self.backbone.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.pos_embed
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x) + self.loras[i](x)
        x = self.backbone.norm(x)
        return self.head(x[:, 0])


class VPTViT(nn.Module):
    def __init__(self, num_classes=102, prompt_len=10, weights_path="./weights/vit_base_patch16_224.pth", require_weights=True):
        super().__init__()
        self.backbone = VisionTransformerBackbone(weights_path=weights_path, require_weights=require_weights)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.prompt_len = prompt_len
        self.prompts = nn.Parameter(torch.randn(1, prompt_len, 768) * 0.02)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        b = x.size(0)
        x = self.backbone.patch_embed(x)
        cls_tokens = self.backbone.cls_token.expand(b, -1, -1)
        prompts = self.prompts.expand(b, -1, -1)
        x = torch.cat((cls_tokens, prompts, x), dim=1)
        # Pad pos embed
        pos = F.pad(self.backbone.pos_embed, (0, 0, 0, self.prompt_len))
        x = x + pos
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        return self.head(x[:, 0])


def build_model(
    model_name: str,
    num_classes: int,
    weights_path: str = "./weights/vit_base_patch16_224.pth",
    use_bca: bool = True,
    use_temp_scale: bool = True,
    require_weights: bool = True,
) -> nn.Module:
    name = model_name.lower().replace("-", "_")
    if name in ["bca_vit", "bcavit", "ours"]:
        return BCAViT(
            num_classes=num_classes,
            weights_path=weights_path,
            use_bca=use_bca,
            use_temp_scale=use_temp_scale,
            require_weights=require_weights,
        )
    elif name in ["linear_probe", "linear"]:
        return LinearProbeViT(num_classes=num_classes, weights_path=weights_path, require_weights=require_weights)
    elif name in ["full_ft", "full"]:
        return FullFineTuneViT(num_classes=num_classes, weights_path=weights_path, require_weights=require_weights)
    elif name == "lora":
        return LoRAViT(num_classes=num_classes, weights_path=weights_path, require_weights=require_weights)
    elif name == "vpt":
        return VPTViT(num_classes=num_classes, weights_path=weights_path, require_weights=require_weights)
    else:
        raise ValueError(f"Unknown model name '{model_name}'.")
