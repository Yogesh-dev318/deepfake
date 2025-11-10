# backend/feature/characterization/audio_saliency.py
import io
import cv2
import torch
import numpy as np
from typing import Tuple
from PIL import Image

from preprocessing.audio_process import ASVSpoofDataset  # reuse your exact preprocessing

def _to_numpy_01(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x.numpy()

def _cv_overlay_from_heat(base_01: np.ndarray, heat_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    base_01: HxW float in [0,1] (grayscale background, e.g., mel/spec)
    heat_01: HxW float in [0,1] (saliency heat)
    returns: HxWx3 uint8 BGR
    """
    base_u8 = (np.clip(base_01, 0, 1) * 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2BGR)

    hm_u8 = (np.clip(heat_01, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(color, alpha, base_rgb, 1 - alpha, 0)
    return out

def _forward_for_shape(model, mel_b, spec_b):
    """
    Tries 3D and 4D shapes; returns (logits, used_shape_str)
    """
    model.eval()
    with torch.no_grad():
        try:
            out = model(mel_b, spec_b, mask_ratio=0.0, do_reconstruct=False)
            return out, "3D (B,F,T)"
        except Exception:
            out = model(mel_b.unsqueeze(1), spec_b.unsqueeze(1), mask_ratio=0.0, do_reconstruct=False)
            return out, "4D (B,1,F,T)"

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

def audio_saliency_png_from_path(
    model,
    audio_path: str,
    device: torch.device,
    alpha: float = 0.45,
    return_mode: str = "triptych"  # 'triptych' | 'mel' | 'spec' | 'combined'
) -> bytes:
    """
    Loads audio via your ASVSpoofDataset pipeline, computes saliency on mel/spec,
    and returns a PNG bytes image (OpenCV-encoded).
    """
    # Reuse your preprocessing exactly
    ds = ASVSpoofDataset([audio_path], [0], augment=False)
    sample = ds[0]
    mel: torch.Tensor = sample["mel"].unsqueeze(0).to(device)   # (1, F, T)
    spec: torch.Tensor = sample["spec"].unsqueeze(0).to(device) # (1, F, T)

    # Normalize bases for visualization (per-sample min-max)
    mel_base = _to_numpy_01(mel[0])
    spec_base = _to_numpy_01(spec[0])

    # Compute gradients
    mel_grad, spec_grad, used_shape, logits = _saliency_grads(model, mel, spec)

    # Convert to [0,1]
    mel_heat = _to_numpy_01(mel_grad[0])
    spec_heat = _to_numpy_01(spec_grad[0])

    # Resize heats to bases if needed (shapes should already match F x T)
    if mel_heat.shape != mel_base.shape:
        mel_heat = cv2.resize(mel_heat, (mel_base.shape[1], mel_base.shape[0]), interpolation=cv2.INTER_LINEAR)
    if spec_heat.shape != spec_base.shape:
        spec_heat = cv2.resize(spec_heat, (spec_base.shape[1], spec_base.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Overlays (grayscale base + heat)
    ov_mel = _cv_overlay_from_heat(mel_base, mel_heat, alpha=alpha)
    ov_spec = _cv_overlay_from_heat(spec_base, spec_heat, alpha=alpha)

    # Combined (average the heats, overlay on mel base by default)
    combined_heat = (mel_heat + spec_heat) / 2.0
    ov_combined = _cv_overlay_from_heat(mel_base, combined_heat, alpha=alpha)

    if return_mode == "triptych":
        canvas = np.concatenate([ov_mel, ov_spec, ov_combined], axis=1)
    elif return_mode == "mel":
        canvas = ov_mel
    elif return_mode == "spec":
        canvas = ov_spec
    elif return_mode == "combined":
        canvas = ov_combined
    else:
        raise ValueError("return_mode must be 'triptych', 'mel', 'spec', or 'combined'")

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
