from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), (H, W)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(F.gelu(self.fc1(x)))
        return self.drop(self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop(y)
        x = x + self.mlp(self.norm2(x))
        return x


class OneStreamViT(nn.Module):
    """Template/search one-stream transformer from scratch."""

    def __init__(self, template_size=128, search_size=256, patch_size=16, in_chans=3, embed_dim=192, depth=8, num_heads=6, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.template_embed = PatchEmbed(template_size, patch_size, in_chans, embed_dim)
        self.search_embed = PatchEmbed(search_size, patch_size, in_chans, embed_dim)
        self.template_grid = template_size // patch_size
        self.search_grid = search_size // patch_size
        self.template_tokens = self.template_grid ** 2
        self.search_tokens = self.search_grid ** 2
        self.pos_template = nn.Parameter(torch.zeros(1, self.template_tokens, embed_dim))
        self.pos_search = nn.Parameter(torch.zeros(1, self.search_tokens, embed_dim))
        self.type_template = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.type_search = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_template, std=0.02)
        nn.init.trunc_normal_(self.pos_search, std=0.02)
        nn.init.trunc_normal_(self.type_template, std=0.02)
        nn.init.trunc_normal_(self.type_search, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, template, search):
        z, _ = self.template_embed(template)
        x, (Hs, Ws) = self.search_embed(search)
        z = z + self.pos_template + self.type_template
        x = x + self.pos_search + self.type_search
        tokens = torch.cat([z, x], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        zt = tokens[:, :self.template_tokens]
        xs = tokens[:, self.template_tokens:]
        fmap = xs.transpose(1, 2).reshape(search.shape[0], -1, Hs, Ws)
        template_token = zt.mean(dim=1)
        search_token = xs.mean(dim=1)
        return fmap, template_token, search_token, zt, xs
