import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads  # 64  多头注意力级联之后的维度，这里有4头注意力
        self.heads = heads  # 4
        self.scale = dim_head ** -0.5  # 16 ** (-0.5) = 0.25
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # (batch, 201, 64 * 3)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),   # 恢复到原始嵌入特征的维度
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # x: (batch, 201, 64)
        b, n, _, h = *x.shape, self.heads  # batch, 201, 64, 4
        
        # get qkv tuple: ((batch, 201, 64), (...), (...))
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v : (batch, 4, 201, 16)
        
        # transpose(k) * q / sqrt(dim_head) -> (batch, head_num, 201, 201)
        dots = torch.einsum('bhid, bhjd->bhij', q, k) * self.scale  # batch matrix multiplication :(batch, 4, 201, 201)
        mask_value = -torch.finfo(dots.dtype).max
            
        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)  # (batch, 4, 201, 201)
        # attn * attention matrix -> output
        out = torch.einsum('bhij, bhjd -> bhid', attn, v)  # (batch, 4, 201, 16)
        # cat all output -> (batch, 201, dim_head * heads)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (batch, 201, 64)
        out = self.to_out(out)  # 恢复到原始嵌入特征的维度以进行残差学习
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, num_channel):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))
            
    def forward(self, x):
        # x: (batch, 201, 64)
        last_output = []
        nl = 0
        for attn, ff in self.layers:
            last_output.append(x)
            if nl > 1:
                x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
            x = attn(x)
            x = ff(x)
            nl += 1
        return x

class ViT(nn.Module):
    """
    dim: 嵌入特征维数 64
    depth: Transformer中编码器的层数(个数) 5
    heads: 多头注意力机制里面的头数 4 
    mlp_dim: Transformer编码器中MLP层的中间隐藏层的维数 8
    dim_head: 注意力的维度
    """
    def __init__(self, img_size=15, in_chans=30, out_chans=64, dim=64, depth=5, heads=4, mlp_dim=16, 
                 dim_head=16, dropout=0.1, emb_dropout=0.1, num_classes=9):
        super(ViT, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_chans, out_chans, 3), nn.BatchNorm2d(64), nn.ReLU())
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 64 + 1, dim))  # (1, 201, dim)，dim表示嵌入特征维度
        self.patch_to_embedding = nn.Linear((img_size - 2) ** 2, dim)  # (batch, 200, 64)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # (1, 1, 64)
        
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, out_chans)
        
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, patch_num, patch*patch*near_band) (batch, 200, 75)
        x = self.conv(x)  # (B, 64, 13, 13)
        x = rearrange(x, 'b c h w -> b c (h w)')  # (B, 64, 169) 
        
        x = self.patch_to_embedding(x)  # (batch, 200, 64)
        b, n, _ = x.shape
        
        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # (batch, 1, 64(dim))
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, 201, 64)
        x += self.pos_embedding # (batch, 201, 64)
        x = self.dropout(x)
        
        # transformer: x: (batch, 201, 64(dim)) -> (batch, 201, 64(dim))
        x = self.transformer(x)
        
        # classification: using cls_token output
        x = self.to_latent(x[:, 0])  # x[:, 0]: (batch, 64)
        
        # MLP classification layer
        return self.mlp_head(x)