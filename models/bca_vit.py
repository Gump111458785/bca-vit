"""Proposed Architecture: Bidirectional Cross-Attention Adapter Vision Transformer (BCA-ViT).
Freezes the pretrained foundation backbone and inserts lightweight bidirectional cross-attention adapters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit_backbone import VisionTransformerBackbone


class BidirectionalCrossAttentionAdapter(nn.Module):
    """Bidirectional Cross-Attention (BCA) Adapter.
    
    Exchanges multi-scale semantic cues between frozen transformer tokens and dynamic task-specific prompts.
    """
    def __init__(self, embed_dim: int = 768, bottleneck_dim: int = 64, num_prompts: int = 8):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.act = nn.GELU()

        # Learnable target domain prompt tokens
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, bottleneck_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(bottleneck_dim, num_heads=4, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        # Down-project tokens
        down = self.act(self.down_proj(x))
        prompts_exp = self.prompts.expand(b, -1, -1)

        # Cross attention between down-projected tokens and prompts
        adapted_prompts, _ = self.cross_attn(prompts_exp, down, down)
        adapted_tokens, _ = self.cross_attn(down, adapted_prompts, adapted_prompts)

        # Up-project and residual add with gating
        out = self.up_proj(adapted_tokens)
        return x + torch.tanh(self.gate) * out


class BCAViT(nn.Module):
    """BCA-ViT: Parameter-Efficient Transfer Model with Pretrained ViT-B/16."""
    def __init__(
        self,
        num_classes: int = 102,
        weights_path: str = "./weights/vit_base_patch16_224.pth",
        use_bca: bool = True,
        use_temp_scale: bool = True,
        require_weights: bool = True,
    ):
        super().__init__()
        self.use_bca = use_bca
        self.use_temp_scale = use_temp_scale

        # Load pretrained backbone
        self.backbone = VisionTransformerBackbone(
            weights_path=weights_path,
            require_weights=require_weights,
        )

        # Freeze foundation backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        embed_dim = 768
        # Insert BCA adapters across blocks
        if use_bca:
            self.adapters = nn.ModuleList([
                BidirectionalCrossAttentionAdapter(embed_dim=embed_dim) for _ in range(12)
            ])
        else:
            self.adapters = None

        self.head = nn.Linear(embed_dim, num_classes)

        if use_temp_scale:
            self.temperature = nn.Parameter(torch.ones(1) * 1.20)
        else:
            self.register_buffer("temperature", torch.ones(1))

    def forward(self, x: torch.Tensor, apply_temp: bool = True) -> torch.Tensor:
        b = x.size(0)
        x = self.backbone.patch_embed(x)
        cls_tokens = self.backbone.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.pos_embed

        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if self.use_bca and self.adapters is not None:
                x = self.adapters[i](x)

        x = self.backbone.norm(x)
        cls_rep = x[:, 0]
        logits = self.head(cls_rep)

        if apply_temp and self.use_temp_scale:
            temp = torch.clamp(self.temperature, min=0.1, max=5.0)
            return logits / temp
        return logits
