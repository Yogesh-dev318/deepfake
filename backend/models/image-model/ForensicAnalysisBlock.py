#ForensicAnalysisBlock.py

import torch
import torch.nn as nn
from typing import Optional


class ForensicAnalysisBlock(nn.Module):
    """
    Forensic Analysis Block (FAB)

    - Path A: 3x3 conv
    - Path B: 5x5 conv
    - Path C: 3x3 dilated conv (dilation=2 or 3)
    - Concat -> 1x1 conv -> BN -> GELU
    - Residual add (projection only if channels mismatch)
    - No final activation after residual
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 2):
        super().__init__()
        if dilation < 1:
            raise ValueError("dilation must be >= 1")

        # Path A: 3x3 conv
        self.path_a = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # Path B: 5x5 conv
        self.path_b = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # Path C: 3x3 dilated conv
        self.path_c = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # Channel-wise concat → reduce via 1x1 conv
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # Residual projection (only if channels mismatch)
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.proj = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for Conv, affine init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)

        a = self.path_a(x)
        b = self.path_b(x)
        c = self.path_c(x)

        concat = torch.cat((a, b, c), dim=1)
        refined = self.refine(concat)

        out = refined + identity
        return out


class ForensicStage(nn.Module):
    """
    Stage = optional strided conv downsample + N FABs.
    """
    def __init__(self,
                 num_blocks: int,
                 in_channels: int,
                 out_channels: int,
                 dilation: int = 2,
                 downsample: bool = False):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        layers = []

        if downsample:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ))
            current_in = out_channels
        else:
            current_in = in_channels

        layers.append(ForensicAnalysisBlock(current_in, out_channels, dilation=dilation))

        for _ in range(num_blocks - 1):
            layers.append(ForensicAnalysisBlock(out_channels, out_channels, dilation=dilation))

        self.stage = nn.Sequential(*layers)

        # # Initialize weights
        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


# # -------------------------
# # shape tests
# # -------------------------
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     fab = ForensicAnalysisBlock(64, 64, dilation=2).to(device)
#     x = torch.randn(2, 64, 128, 128, device=device)
#     y = fab(x)
#     print("FAB (64->64):", y.shape)  # expect [2, 64, 128, 128]

#     fab_proj = ForensicAnalysisBlock(64, 128, dilation=3).to(device)
#     y2 = fab_proj(x)
#     print("FAB (64->128):", y2.shape)  # expect [2, 128, 128, 128]

#     stage = ForensicStage(3, 64, 128, dilation=2, downsample=True).to(device)
#     y3 = stage(x)
#     print("Stage (downsample):", y3.shape)  # expect [2, 128, 64, 64]

#     stage2 = ForensicStage(2, 128, 128, dilation=2, downsample=False).to(device)
#     y4 = stage2(y3)
#     print("Stage (no downsample):", y4.shape)  # expect [2, 128, 64, 64]
