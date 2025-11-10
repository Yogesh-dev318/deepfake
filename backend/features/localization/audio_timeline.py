# backend/feature/characterization/audio_saliency.py
import io
import cv2
import torch
import numpy as np
from typing import Tuple
from PIL import Image

from preprocessing.audio_process import ASVSpoofDataset  # reuse your exact preprocessing

# def _to_numpy_01(x: torch.Tensor) -> np.ndarray:
#     x = x.detach().cpu().float()
#     x = x - x.min()
#     if x.max() > 0:
#         x = x / x.max()
#     return x.numpy()

# def _cv_overlay_from_heat(base_01: np.ndarray, heat_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
#     """
#     base_01: HxW float in [0,1] (grayscale background, e.g., mel/spec)
#     heat_01: HxW float in [0,1] (saliency heat)
#     returns: HxWx3 uint8 BGR
#     """
#     base_u8 = (np.clip(base_01, 0, 1) * 255).astype(np.uint8)
#     base_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2BGR)

#     hm_u8 = (np.clip(heat_01, 0, 1) * 255).astype(np.uint8)
#     color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
#     out = cv2.addWeighted(color, alpha, base_rgb, 1 - alpha, 0)
#     return out

# def _forward_for_shape(model, mel_b, spec_b):
#     """
#     Tries 3D and 4D shapes; returns (logits, used_shape_str)
#     """
#     model.eval()
#     with torch.no_grad():
#         try:
#             out = model(mel_b, spec_b, mask_ratio=0.0, do_reconstruct=False)
#             return out, "3D (B,F,T)"
#         except Exception:
#             out = model(mel_b.unsqueeze(1), spec_b.unsqueeze(1), mask_ratio=0.0, do_reconstruct=False)
#             return out, "4D (B,1,F,T)"

def _saliency_grads(model, mel_b: torch.Tensor, spec_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
    """
    Computes |d logit_spoof / d mel| and |d logit_spoof / d spec|
    Returns: (mel_grad_abs, spec_grad_abs, used_shape, logits)
    """
    model.train(False)  # eval graph, but gradients on inputs

    mel_b = mel_b.clone().detach().requires_grad_(True)
    spec_b = spec_b.clone().detach().requires_grad_(True)

    # Forward with gradients (no torch.no_grad here!)
    try:
        logits, used_shape = model(mel_b, spec_b, mask_ratio=0.0, do_reconstruct=False), "3D (B,F,T)"
    except Exception:
        logits, used_shape = model(mel_b.unsqueeze(1), spec_b.unsqueeze(1), mask_ratio=0.0, do_reconstruct=False), "4D (B,1,F,T)"

    # Your DBMFC returns (logits, emb, ...) per app.py
    if isinstance(logits, tuple):
        logits = logits[0]  # (B,2)

    # Backprop w.r.t. spoof class = index 1
    target = logits[:, 1].sum()
    model.zero_grad(set_to_none=True)
    if mel_b.grad is not None: mel_b.grad.zero_()
    if spec_b.grad is not None: spec_b.grad.zero_()

    target.backward(retain_graph=False)

    mg = mel_b.grad.abs()
    sg = spec_b.grad.abs()
    return mg, sg, used_shape, logits.detach()

# def audio_saliency_png_from_path(
#     model,
#     audio_path: str,
#     device: torch.device,
#     alpha: float = 0.45,
#     return_mode: str = "triptych"  # 'triptych' | 'mel' | 'spec' | 'combined'
# ) -> bytes:
#     """
#     Loads audio via your ASVSpoofDataset pipeline, computes saliency on mel/spec,
#     and returns a PNG bytes image (OpenCV-encoded).
#     """
#     # Reuse your preprocessing exactly
#     ds = ASVSpoofDataset([audio_path], [0], augment=False)
#     sample = ds[0]
#     mel: torch.Tensor = sample["mel"].unsqueeze(0).to(device)   # (1, F_mel, T)
#     spec: torch.Tensor = sample["spec"].unsqueeze(0).to(device) # (1, F_spec, T)

#     # Normalize bases for visualization (per-sample min-max)
#     mel_base = _to_numpy_01(mel[0])   # (F_mel, T)
#     spec_base = _to_numpy_01(spec[0]) # (F_spec, T)

#     # Compute gradients
#     mel_grad, spec_grad, used_shape, logits = _saliency_grads(model, mel, spec)

#     # Convert to [0,1]
#     mel_heat = _to_numpy_01(mel_grad[0])   # (F_mel, T)
#     spec_heat = _to_numpy_01(spec_grad[0]) # (F_spec, T)

#     # If shapes differ, resize heats to match their bases (safety)
#     if mel_heat.shape != mel_base.shape:
#         mel_heat = cv2.resize(
#             mel_heat,
#             (mel_base.shape[1], mel_base.shape[0]),  # (T, F_mel)
#             interpolation=cv2.INTER_LINEAR
#         )
#     if spec_heat.shape != spec_base.shape:
#         spec_heat = cv2.resize(
#             spec_heat,
#             (spec_base.shape[1], spec_base.shape[0]),  # (T, F_spec)
#             interpolation=cv2.INTER_LINEAR
#         )

#     # Overlays (grayscale base + heat)
#     ov_mel = _cv_overlay_from_heat(mel_base, mel_heat, alpha=alpha)
#     ov_spec = _cv_overlay_from_heat(spec_base, spec_heat, alpha=alpha)

#     # --- FIX 1: make combined heat on mel resolution ---
#     # Map spec heat onto mel's (F_mel, T) grid before averaging
#     spec_heat_on_mel = cv2.resize(
#         spec_heat,
#         (mel_base.shape[1], mel_base.shape[0]),  # (T, F_mel)
#         interpolation=cv2.INTER_AREA             # safe downsampling (e.g., 1025 -> 80)
#     )
#     combined_heat = (mel_heat + spec_heat_on_mel) / 2.0
#     ov_combined = _cv_overlay_from_heat(mel_base, combined_heat, alpha=alpha)

#     # Return single-pane modes directly
#     if return_mode == "mel":
#         canvas = ov_mel
#     elif return_mode == "spec":
#         canvas = ov_spec
#     elif return_mode == "combined":
#         canvas = ov_combined
#     elif return_mode == "triptych":
#         # --- FIX 2: normalize overlay heights before concatenation ---
#         target_h = max(ov_mel.shape[0], ov_spec.shape[0], ov_combined.shape[0])

#         def _resize_h(img, H):
#             h, w = img.shape[:2]
#             if h == H:
#                 return img
#             new_w = int(round(w * (H / h)))
#             return cv2.resize(img, (new_w, H), interpolation=cv2.INTER_LINEAR)

#         ov_mel_r = _resize_h(ov_mel, target_h)
#         ov_spec_r = _resize_h(ov_spec, target_h)
#         ov_combined_r = _resize_h(ov_combined, target_h)

#         canvas = np.concatenate([ov_mel_r, ov_spec_r, ov_combined_r], axis=1)
#     else:
#         raise ValueError("return_mode must be 'triptych', 'mel', 'spec', or 'combined'")

#     ok, buf = cv2.imencode(".png", canvas)
#     if not ok:
#         raise RuntimeError("PNG encoding failed")
#     return buf.tobytes()


import numpy as np
import cv2
import torch

def _minmax01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    mx = x.max()
    return x / mx if mx > 0 else x

def _to_db_01(x: np.ndarray, eps: float = 1e-8, pclip: float = 99.5) -> np.ndarray:
    """
    Convert linear magnitude to log-dB-ish scale, then percentile clip and min-max to [0,1].
    Works for mel and spec equally well.
    """
    x = np.log1p(np.maximum(x, 0.0) + eps)
    # percentile clip for better contrast
    hi = np.percentile(x, pclip)
    lo = np.percentile(x, 100 - pclip)
    x = np.clip(x, lo, hi)
    return _minmax01(x)

def _clahe_01(gray01: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Optional contrast enhancement. Input/Output in [0,1].
    """
    gray = (np.clip(gray01, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    out = clahe.apply(gray)
    return out.astype(np.float32) / 255.0

def _heat_smooth_01(heat01: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Optional gentle Gaussian blur to reduce speckle. Keeps [0,1].
    """
    if ksize and ksize >= 3:
        heat = cv2.GaussianBlur(heat01, (ksize, ksize), 0)
        return _minmax01(heat)
    return heat01

def _colorize_heat(heat01: np.ndarray) -> np.ndarray:
    """
    Convert [0,1] heatmap to BGR using COLORMAP_TURBO (better than JET).
    """
    hm = (np.clip(heat01, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(hm, cv2.COLORMAP_TURBO)

def _overlay_colored(base01: np.ndarray, heat01: np.ndarray, alpha: float) -> np.ndarray:
    """
    base01: HxW [0,1] grayscale
    heat01: HxW [0,1]
    Returns BGR uint8 overlay.
    """
    base_u8 = (np.clip(base01, 0, 1) * 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2BGR)
    color = _colorize_heat(heat01)
    out = cv2.addWeighted(color, alpha, base_bgr, 1 - alpha, 0)
    return out

def _draw_hot_contours(overlay_bgr: np.ndarray, heat01: np.ndarray, thresh: float = 0.6, thickness: int = 1) -> np.ndarray:
    """
    Draw contour lines around the hottest regions to make focus areas explicit.
    """
    mask = (heat01 >= thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_bgr, contours, -1, (255, 255, 255), thickness)  # white outline
    return overlay_bgr

def _append_time_strip(panel_bgr: np.ndarray, heat01: np.ndarray, strip_h: int = 28) -> np.ndarray:
    """
    Append a small bar under the panel showing time-importance (sum over frequency).
    """
    H, W = panel_bgr.shape[:2]
    tcurve = heat01.mean(axis=0)  # [T]
    tcurve = _minmax01(tcurve)
    strip = np.full((strip_h, W), 30, dtype=np.uint8)  # dark background
    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)

    # Draw curve
    xs = np.linspace(0, W - 1, num=len(tcurve)).astype(int)
    ys = (strip_h - 1 - (tcurve * (strip_h - 6))).astype(int)  # padding
    pts = np.vstack([xs, ys]).T.reshape(-1, 1, 2)
    cv2.polylines(strip_bgr, [pts], isClosed=False, color=(0, 255, 255), thickness=2)  # cyan curve
    # Merge
    return np.concatenate([panel_bgr, strip_bgr], axis=0)

def _resize_same_height(imgs, interp=cv2.INTER_LINEAR):
    target_h = max(im.shape[0] for im in imgs)
    out = []
    for im in imgs:
        if im.shape[0] == target_h:
            out.append(im)
        else:
            scale = target_h / im.shape[0]
            new_w = int(round(im.shape[1] * scale))
            out.append(cv2.resize(im, (new_w, target_h), interpolation=interp))
    return out


def audio_saliency_png_from_path(
    model,
    audio_path: str,
    device: torch.device,
    alpha: float = 0.45,
    return_mode: str = "triptych",   # 'triptych' | 'mel' | 'spec' | 'combined'
    enhance_base: bool = True,       # log-dB + percentile clip + CLAHE
    clahe: bool = True,
    blur_heat_ksize: int = 3,        # 0 or None to disable
    draw_contours: bool = True,      # outline hottest regions
    contour_thresh: float = 0.6      # threshold in [0,1]
) -> bytes:
    """
    Produces higher-quality, more interpretable overlays:
      - Better base rendering (log-dB + contrast)
      - Smoothed heat
      - Contour outlines
      - Time-importance strip
    """
    # Load with your exact preprocessing
    ds = ASVSpoofDataset([audio_path], [0], augment=False)
    sample = ds[0]
    mel: torch.Tensor  = sample["mel"].unsqueeze(0).to(device)   # (1, F_mel, T)
    spec: torch.Tensor = sample["spec"].unsqueeze(0).to(device)  # (1, F_spec, T)

    # Keep copies for visualization bases before grads (convert to numpy)
    mel_base = mel.detach().cpu().numpy()[0]   # (F_mel, T)
    spec_base = spec.detach().cpu().numpy()[0] # (F_spec, T)

    # Convert bases to visually clear grayscale (0..1)
    if enhance_base:
        mel_base01  = _to_db_01(mel_base)
        spec_base01 = _to_db_01(spec_base)
        if clahe:
            mel_base01  = _clahe_01(mel_base01)
            spec_base01 = _clahe_01(spec_base01)
    else:
        # simple min-max
        mel_base01  = _minmax01(mel_base)
        spec_base01 = _minmax01(spec_base)

    # Compute input-gradient saliency
    mel_grad, spec_grad, _, _ = _saliency_grads(model, mel, spec)

    # To [0,1]
    mel_heat = _minmax01(mel_grad[0].detach().cpu().numpy())    # (F_mel, T)
    spec_heat = _minmax01(spec_grad[0].detach().cpu().numpy())  # (F_spec, T)

    # Optional smoothing to reduce speckle
    if blur_heat_ksize and blur_heat_ksize >= 3:
        mel_heat  = _heat_smooth_01(mel_heat,  blur_heat_ksize)
        spec_heat = _heat_smooth_01(spec_heat, blur_heat_ksize)

    # Overlays (ensure same shapes for base and heat)
    if mel_heat.shape != mel_base01.shape:
        mel_heat = cv2.resize(mel_heat, (mel_base01.shape[1], mel_base01.shape[0]), interpolation=cv2.INTER_LINEAR)
    if spec_heat.shape != spec_base01.shape:
        spec_heat = cv2.resize(spec_heat, (spec_base01.shape[1], spec_base01.shape[0]), interpolation=cv2.INTER_LINEAR)

    ov_mel = _overlay_colored(mel_base01, mel_heat, alpha=alpha)
    ov_spec = _overlay_colored(spec_base01, spec_heat, alpha=alpha)

    # Optional contours to make hotspots pop
    if draw_contours:
        ov_mel  = _draw_hot_contours(ov_mel,  mel_heat,  thresh=contour_thresh, thickness=1)
        ov_spec = _draw_hot_contours(ov_spec, spec_heat, thresh=contour_thresh, thickness=1)

    # Combined map on mel resolution
    spec_heat_on_mel = cv2.resize(spec_heat, (mel_base01.shape[1], mel_base01.shape[0]), interpolation=cv2.INTER_AREA)
    combined_heat = _minmax01(0.5 * (mel_heat + spec_heat_on_mel))
    ov_combined = _overlay_colored(mel_base01, combined_heat, alpha=alpha)
    if draw_contours:
        ov_combined = _draw_hot_contours(ov_combined, combined_heat, thresh=contour_thresh, thickness=1)

    # Append time-importance strips
    ov_mel  = _append_time_strip(ov_mel,  mel_heat)
    ov_spec = _append_time_strip(ov_spec, spec_heat)
    ov_combined = _append_time_strip(ov_combined, combined_heat)

    # Return mode
    if return_mode == "mel":
        canvas = ov_mel
    elif return_mode == "spec":
        canvas = ov_spec
    elif return_mode == "combined":
        canvas = ov_combined
    elif return_mode == "triptych":
        ov_mel_r, ov_spec_r, ov_combined_r = _resize_same_height([ov_mel, ov_spec, ov_combined])
        canvas = np.concatenate([ov_mel_r, ov_spec_r, ov_combined_r], axis=1)
    else:
        raise ValueError("return_mode must be 'triptych', 'mel', 'spec', or 'combined'")

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
