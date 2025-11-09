#AdaptiveFrequencyAttentionModule.py


import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWPT2D(nn.Module):
    """Two-level Haar Wavelet Packet Transform (non-learnable)."""
    def __init__(self):
        super().__init__()
        h = torch.tensor([1.0, 1.0]) / 2**0.5
        g = torch.tensor([1.0, -1.0]) / 2**0.5

        ll = torch.einsum("i,j->ij", h, h)
        lh = torch.einsum("i,j->ij", h, g)
        hl = torch.einsum("i,j->ij", g, h)
        hh = torch.einsum("i,j->ij", g, g)
        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)

        self.register_buffer("kernels", kernels)

    def forward(self, x: torch.Tensor) -> list:
        b, c, h, w = x.shape
        weight = self.kernels.repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, groups=c)  # (B,4C,H/2,W/2)
        return torch.chunk(y, 4, dim=1)  # [LL,LH,HL,HH]


class SharedSubbandCNN(nn.Module):
    def __init__(self, in_ch, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU()
        )

    def forward(self, x): 
        return self.net(x)


class AFAM(nn.Module):
    """Adaptive Frequency Attention Module with projector for fusion."""
    def __init__(self, in_ch=3, sub_cnn_out=64, attn_hidden=32, proj_ch=128):
        super().__init__()
        self.wpt = HaarWPT2D()
        self.sub_cnn = SharedSubbandCNN(in_ch, sub_cnn_out)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(sub_cnn_out, attn_hidden),
            nn.GELU(),
            nn.Linear(attn_hidden, 1),
            nn.Sigmoid()
        )
        self.projector = nn.Conv2d(sub_cnn_out, proj_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, hf_bands_override: list = None) -> torch.Tensor:
        if hf_bands_override is None:
            l1_bands = self.wpt(x)
            l2_bands = self.wpt(l1_bands[0])  # decompose LL again
            hf_bands = list(l1_bands[1:]) + list(l2_bands[1:])
        else:
            hf_bands = hf_bands_override

        feats = []
        for sb in hf_bands:
            f = self.sub_cnn(sb)
            g = self.gap(f).flatten(1)
            w = self.mlp(g)
            feats.append(f * w.view(-1, 1, 1, 1))

        target_size = hf_bands[0].shape[2:]
        resized = [f if f.shape[2:] == target_size else 
                   F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                   for f in feats]

        fused = torch.stack(resized, dim=0).sum(0)
        fused = self.projector(fused)
        return fused



# # -------------------------
# # test
# # -------------------------
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = AFAM(in_ch=3, sub_cnn_out=64, attn_hidden=32, proj_ch=128).to(device)

#     x = torch.randn(2, 3, 128, 128, device=device)  # RGB image
#     y = model(x)

#     print("Input:", x.shape)
#     print("Output (fused frequency feature):", y.shape)  # expect (2,128,64,64)

#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print("Total trainable params:", total_params)          # 86,145
