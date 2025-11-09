# src/models/BSFNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.imageModel.ForensicAnalysisBlock import ForensicAnalysisBlock, ForensicStage
from models.imageModel.AdaptiveFrequencyAttentionModule import AFAM
from models.imageModel.BidirectionalGuidedAttention import BidirectionalGuidedAttention


class Stem(nn.Module):
    
    """Spatial stem: 7x7 Conv -> BN -> GELU (stride=2)."""
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Downsample(nn.Module):
    """Generic 3x3 stride-2 downsample conv -> BN -> GELU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def init_weights(m):
    """Centralized weight init."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


class BSFNet(nn.Module):
    """Four-stage BSF-Net (dual-stream with bidirectional fusion)."""
    def __init__(self, in_ch=3, base_ch=32, num_classes=1):
        super().__init__()

        # Stem (Stage 0)
        self.spatial_stem = Stem(in_ch=in_ch, out_ch=base_ch)

        # Stage 1
        self.stage1_spatial = ForensicStage(num_blocks=2, in_channels=base_ch, out_channels=base_ch)
        afam1 = AFAM(in_ch=in_ch, sub_cnn_out=64, attn_hidden=32, proj_ch=base_ch)
        self.stage1_fusion = BidirectionalGuidedAttention(afam_module=afam1, spatial_ch=base_ch, freq_proj_ch=base_ch)
        self.stage1_down_s = Downsample(base_ch, base_ch * 2)
        self.stage1_down_f = Downsample(base_ch, base_ch * 2)

        # Stage 2
        ch2 = base_ch * 2
        self.stage2_spatial = ForensicStage(num_blocks=3, in_channels=ch2, out_channels=ch2)
        afam2 = AFAM(in_ch=in_ch, sub_cnn_out=64, attn_hidden=32, proj_ch=ch2)
        self.stage2_fusion = BidirectionalGuidedAttention(afam_module=afam2, spatial_ch=ch2, freq_proj_ch=ch2)
        self.stage2_down_s = Downsample(ch2, ch2 * 2)
        self.stage2_down_f = Downsample(ch2, ch2 * 2)

        # Stage 3
        ch3 = ch2 * 2
        self.stage3_spatial = ForensicStage(num_blocks=4, in_channels=ch3, out_channels=ch3)
        afam3 = AFAM(in_ch=in_ch, sub_cnn_out=64, attn_hidden=32, proj_ch=ch3)
        self.stage3_fusion = BidirectionalGuidedAttention(afam_module=afam3, spatial_ch=ch3, freq_proj_ch=ch3)
        self.stage3_down_s = Downsample(ch3, ch3 * 2)
        self.stage3_down_f = Downsample(ch3, ch3 * 2)

        # Stage 4 (no downsampling)
        ch4 = ch3 * 2
        self.stage4_spatial = ForensicStage(num_blocks=2, in_channels=ch4, out_channels=ch4)
        afam4 = AFAM(in_ch=in_ch, sub_cnn_out=64, attn_hidden=32, proj_ch=ch4)
        self.stage4_fusion = BidirectionalGuidedAttention(afam_module=afam4, spatial_ch=ch4, freq_proj_ch=ch4)

        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(ch4 * 2, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        B = x.size(0)

        # Stage 0
        s0 = self.spatial_stem(x)

        # Stage 1
        s1_pre = self.stage1_spatial(s0)
        s1_post, f1, _ = self.stage1_fusion(x, s1_pre)
        s1_out = self.stage1_down_s(s1_post)
        f1_out = self.stage1_down_f(f1)

        # Stage 2
        s2_pre = self.stage2_spatial(s1_out)
        s2_post, f2, _ = self.stage2_fusion(x, s2_pre)
        s2_out = self.stage2_down_s(s2_post)
        f2_out = self.stage2_down_f(f2)

        # Stage 3
        s3_pre = self.stage3_spatial(s2_out)
        s3_post, f3, _ = self.stage3_fusion(x, s3_pre)
        s3_out = self.stage3_down_s(s3_post)
        f3_out = self.stage3_down_f(f3)

        # Stage 4
        s4_pre = self.stage4_spatial(s3_out)
        s4_post, f4, _ = self.stage4_fusion(x, s4_pre)

        # Classification
        s_vec = self.gap(s4_post).view(B, -1)
        f_vec = self.gap(f4).view(B, -1)
        fused_vec = torch.cat([s_vec, f_vec], dim=1)
        out = self.classifier(fused_vec)
        return out