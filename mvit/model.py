import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np
from timm.models.vision_transformer import Block

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M, )
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2, )  32

    pos = pos.reshape(-1)  # (M, )  169
    out = np.einsum('m, d -> md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)  # (1+H*W, D)
    return pos_embed

class MViT(nn.Module):
    def __init__(self, in_chans=1, bands=30, num_classes=9, dim=64, heads=4, depth=3, dropout=0.2):
        super(MViT, self).__init__()
        self.conv3d = nn.Sequential(nn.Conv3d(1, 8, 3), nn.BatchNorm3d(8), nn.ReLU())
        self.conv2d = nn.Sequential(nn.Conv2d(8*28, 64, 3), nn.BatchNorm2d(64), nn.ReLU())

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls_token, std=.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, 121 + 1, dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(dim, 11, cls_token=True) 
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.blocks = nn.ModuleList([Block(dim, heads, qkv_bias=True, attn_drop=0.1, drop=0.1) for _ in range(depth)])

        self.norm = nn.LayerNorm(dim)
        
        self.cls_head = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def random_masking(self, x, mask_ratio=0.75):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]  (N, L)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # 从小到大排序，返回索引 (N, L)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def forward(self, x, mask_ratio=0.75):
        # (B, 1, 30, 15, 15)
        x = self.conv3d(x)  # (B, 8, 28, 13, 13)
        x = rearrange(x, 'b c d h w -> b (c d) h w')  # (B, 224, 13, 13)
        x = self.conv2d(x)  # (B, 64, 11, 11)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, 121, 64) 

        x = x + self.pos_embed[:, 1:, :]

        center_embed = x[:, 60, :].unsqueeze(1)  # (B, 1, 64)
        x = torch.cat([x[:, :60, :], x[:, 61:, :]], dim=1)  # (B, 120, 64)
        
        if self.training:
            x = self.random_masking(x, mask_ratio)  # (B, 30, 64)

        x = torch.cat([center_embed, x], dim=1)

        cls_token = self.cls_token + self.pos_embed[:, :1, :] 
        cls_token = repeat(cls_token, '1 n d -> b n d', b = x.shape[0])  # (B, 1, 64)
        
        x = torch.cat((cls_token, x), dim = 1)

        x = self.dropout(x)  # (B, 31, 64)

        for blk in self.blocks:
            x = blk(x)

        latent = self.norm(x[:, 0, :])
        x = self.cls_head(latent)
        return x