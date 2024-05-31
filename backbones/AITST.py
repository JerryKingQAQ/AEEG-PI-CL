# -*- coding = utf-8 -*-
# @File : Interrelated Temporal-Spatial Transformer.py
# @Software : PyCharm

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Spatial_Attention(nn.Module):
    def __init__(self, channels, channel_rate=1):
        super().__init__()
        mid_channels = channels // channel_rate

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # spatial attention
        x_spatial_att = self.spatial_attention(x).sigmoid()
        x = x * x_spatial_att
        return x


class SA_Encoder(nn.Module):
    def __init__(self, dim, depth, channels, channel_rate=1, mlp_dim=256, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Spatial_Attention(channels=channels, channel_rate=channel_rate)),
                PreNorm(dim, FeedForward(dim[-1], mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for sa, ff in self.layers:
            x = sa(x) + x
            x = ff(x) + x
        return x


class Channel_Attention(nn.Module):
    def __init__(self, channels, channel_rate=1):
        super().__init__()

        mid_channels = channels // channel_rate

        self.channel_attention = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # channel attention
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        return x


class CA_Encoder(nn.Module):
    def __init__(self, dim, depth, channels, channel_rate=1, mlp_dim=256, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Channel_Attention(channels=channels, channel_rate=channel_rate)),
                PreNorm(dim, FeedForward(dim[-1], mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for ca, ff in self.layers:
            x = ca(x) + x
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class AITST(nn.Module):
    def __init__(self, input_size, patch_size,channels, num_classes, model_dim,
                 emb_dropout=0.1, pool='cls', dim_head=256, channel_rate=5,
                 depth=8, heads=16, mlp_dim=512, dropout=0., model_structure="sa+ca"):
        super(AITST, self).__init__()

        input_height, input_width = input_size[-2:]
        patch_height, patch_width = patch_size
        num_patches = (input_height // patch_height) * (input_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_size = patch_size
        self.num_classes = num_classes

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, model_dim),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.weight = torch.randn(patch_dim, model_dim).to(DEVICE)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, model_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8).to(DEVICE)

        self.transformer = Transformer(model_dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.SA_Encoder = SA_Encoder(dim=input_size, depth=2, channels=channels, channel_rate=channel_rate,
                                     dropout=0.1)
        self.CA_Encoder = CA_Encoder(dim=input_size, depth=2, channels=channels, channel_rate=channel_rate,
                                     dropout=0.1)
        self.model_structure = model_structure

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_classes)
        )
        self.out_dim = model_dim

    def forward(self, x):
        if self.model_structure == "sa+ca":
            # spatial_attention
            x = self.SA_Encoder(x)
            # channel attention
            x = self.CA_Encoder(x)
        elif self.model_structure == "sa":
            # spatial_attention
            x = self.SA_Encoder(x)
        elif self.model_structure == "ca":
            # channel attention
            x = self.CA_Encoder(x)
        elif self.model_structure == "no+sa+ca":
            pass

        # step 1 convert input to embedding vector sequence
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # step 2 prepend CLS token embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # step3 add position embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # step4 pass embedding to Transformer Encoder
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # step5 do classification
        x = self.to_latent(x)

        return {
            'features': x
        }

if __name__ == '__main__':
    x = torch.randn((8, 32, 375, 6))
    model = AITST(input_size=(32,375,6), patch_size=(15,1),channels=32, num_classes=10, model_dim=256,
                 emb_dropout=0.3, pool='cls', dim_head=256, channel_rate=4,
                 depth=4, heads=2, mlp_dim=256, dropout=0.3, model_structure="sa+ca")
    out = model(x)
    # print(out.shape)
    print(out['features'].shape)
