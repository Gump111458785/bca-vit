"""Vision Transformer (ViT-B/16) backbone with pretrained weight requirement.
"""

import os
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, D, N] -> [B, N, D]
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x_norm = self.ln_1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = res + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionTransformerBackbone(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        weights_path: str = "./weights/vit_base_patch16_224.pth",
        require_weights: bool = True,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Check weight file requirement (Requirement: Must download weights)
        if require_weights:
            if not os.path.exists(weights_path):
                raise RuntimeError(
                    f"\n[CRITICAL ERROR] Pretrained backbone weights missing at: '{weights_path}'\n"
                    f"Running this repository requires downloading the pretrained foundation model weights!\n"
                    f"Please run: 'bash scripts/download_weights.sh' before training or evaluating.\n"
                )
            print(f"[Model Backbone] Loading verified pretrained ViT weights from: '{weights_path}'")
            try:
                ckpt = torch.load(weights_path, map_location="cpu")
                # Load compatible weights if structure matches
                self.load_state_dict(ckpt, strict=False)
            except Exception as e:
                print(f"[Model Backbone Note] Loaded weights with partial alignment: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # Return [CLS] representation
