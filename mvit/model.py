import torch
import torch.nn as nn
from einops import rearrange, repeat


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