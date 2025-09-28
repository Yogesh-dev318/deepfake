# BidirectionalGuidedAttention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FGSA(nn.Module):
    """
    Frequency-Guided Spatial Attention (FGSA)
    Two 1x1 convs -> GlobalAvgPool -> Sigmoid -> channel-wise attention
    """
    def __init__(self, freq_ch: int, spatial_ch: int, mid_ch: int = None):
        super().__init__()
        mid = mid_ch if mid_ch is not None else max(freq_ch, spatial_ch)
        self.conv1 = nn.Conv2d(freq_ch, mid, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(mid, spatial_ch, kernel_size=1, bias=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, freq_feat: torch.Tensor, spatial_feat: torch.Tensor):
        # freq_feat: (B, F_ch, Hf, Wf)
        # spatial_feat: (B, S_ch, Hs, Ws)
        B, S_ch, Hs, Ws = spatial_feat.shape

        if freq_feat.shape[2:] != (Hs, Ws):
            freq = F.interpolate(freq_feat, size=(Hs, Ws), mode="bilinear", align_corners=False)
        else:
            freq = freq_feat

        x = self.conv1(freq)
        x = self.conv2(x)                 # (B, S_ch, Hs, Ws)
        x = self.gap(x)                   # (B, S_ch, 1, 1)
        att = self.sigmoid(x)             # (B, S_ch, 1, 1)
        modulated = spatial_feat * att    # broadcast over H/W
        att_vec = att.view(B, S_ch)       # (B, S_ch)
        return modulated, att_vec


class SGFA(nn.Module):
    """
    Spatial-Guided Frequency Attention (SGFA)
    1x1 conv on spatial features -> Sigmoid => spatial heatmap (B,1,H,W)
    Used to weight raw HF subbands (list of tensors) before AFAM processes them.
    """
    def __init__(self, spatial_ch: int):
        super().__init__()
        self.head = nn.Conv2d(spatial_ch, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, spatial_feat: torch.Tensor, hf_bands: List[torch.Tensor]) -> List[torch.Tensor]:
        # spatial_feat: (B, S_ch, Hs, Ws)
        att_map = self.sigmoid(self.head(spatial_feat))  # (B,1,Hs,Ws)
        weighted_bands = []
        for sb in hf_bands:
            target_size = sb.shape[2:]
            if att_map.shape[2:] != target_size:
                att_resized = F.interpolate(att_map, size=target_size, mode="bilinear", align_corners=False)
            else:
                att_resized = att_map
            weighted_bands.append(sb * att_resized)  # broadcast across channels
        return weighted_bands


class BidirectionalGuidedAttention(nn.Module):
    """
    Wrapper that coordinates SGFA -> AFAM -> FGSA.
    - afam_module must expose:
        * a .wpt member (callable) or this class will call afam_module.wpt(x)
        * a forward(x, hf_bands_override: list = None) signature that accepts hf_bands_override
    """
    def __init__(self, afam_module: nn.Module, spatial_ch: int, freq_proj_ch: int, fgsa_mid: int = None):
        super().__init__()
        if not hasattr(afam_module, "wpt"):
            raise ValueError("afam_module must provide a .wpt attribute (WPT operator).")
        self.afam = afam_module
        self.sgfa = SGFA(spatial_ch=spatial_ch)
        self.fgsa = FGSA(freq_ch=freq_proj_ch, spatial_ch=spatial_ch, mid_ch=fgsa_mid)

    def _extract_hf_bands(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Uses AFAM's WPT instance to get level1 and level2 bands
        l1 = self.afam.wpt(x)                 # [LL1,LH1,HL1,HH1]
        l2 = self.afam.wpt(l1[0])             # decompose LL1 -> [LL2,LH2,HL2,HH2]
        hf = list(l1[1:]) + list(l2[1:])      # LH1,HL1,HH1, LH2,HL2,HH2
        return hf

    def forward(self, img: torch.Tensor, spatial_feat: torch.Tensor):
        """
        Args:
            img: input image tensor (B,3,H,W)
            spatial_feat: feature map from a FAB (B, S_ch, Hs, Ws)
        Returns:
            modulated_spatial: spatial feature modulated by FGSA
            fused_freq: fused frequency feature from AFAM (B, Fproj, Hf, Wf)
            att_vec: channel-wise attention vector produced by FGSA (B, S_ch)
        """

        hf_bands = self._extract_hf_bands(img)
        weighted_hf = self.sgfa(spatial_feat, hf_bands)
        fused_freq = self.afam.forward(img, hf_bands_override=weighted_hf)
        modulated_spatial, att_vec = self.fgsa(fused_freq, spatial_feat)

        return modulated_spatial, fused_freq, att_vec
