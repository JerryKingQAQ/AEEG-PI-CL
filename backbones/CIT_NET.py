import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


class Conv_Layers(nn.Module):
    def __init__(self, in_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 4, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 4, in_channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 10, patch_size_x: int = 3, patch_size_y: int = 3, emb_size: int = 90,
                 img_size: int = 9):
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.positions_num = (img_size[0] // self.patch_size_x) * (img_size[1] // self.patch_size_y) + 1
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=(patch_size_x, patch_size_y),
                      stride=(patch_size_x, patch_size_y)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(self.positions_num, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 90, num_heads: int = 9):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling

        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4):  # expansion上采样
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 90,
                 forward_expansion: int = 4,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, **kwargs),
                nn.LayerNorm(emb_size)
            )),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion),
                nn.LayerNorm(emb_size)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 90, out_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.head = nn.Sequential(Reduce('b n e -> b e', reduction='mean'),
                                  nn.LayerNorm(emb_size),
                                  nn.Linear(emb_size, out_dim))
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = self.head(x)
        return {
            'features': x
        }


class CIT_NET(nn.Sequential):
    def __init__(self,
                 in_channels: int = 10,
                 patch_size_x: int = 3,
                 patch_size_y: int = 3,
                 emb_size: int = 90,
                 img_size: int = 9,
                 depth: int = 12,
                 out_dim: int = 256,  # SEED-V: 5; SEED-IV: 4
                 **kwargs):
        super().__init__(
            Conv_Layers(in_channels=in_channels),
            PatchEmbedding(in_channels, patch_size_x, patch_size_y, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, out_dim)
        )
        self.out_dim = out_dim

if __name__ == '__main__':
    x = torch.randn((8, 32, 375, 6))
    model = CIT_NET(in_channels=32, patch_size_x=15, patch_size_y=6, img_size=(375, 6))
    out = model(x)
    print(out['features'].shape)
