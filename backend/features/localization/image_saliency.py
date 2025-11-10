# backend/characterization/image_saliency.py
import io
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

# ---------- Grad-CAM for BSFNet (dual-stream at stage4_fusion) ----------

class GradCAM_BSFNet:
    """
    Hooks BSFNet.stage4_fusion to produce two CAMs:
    - spatial stream (s4_post)
    - frequency stream (f4)
    Optional combined CAM (avg or max).
    """
    def __init__(self, model, target_module=None, device=None):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.target_module = target_module or getattr(model, "stage4_fusion")

        self._fmap_s = None
        self._fmap_f = None
        self._gmap_s = None
        self._gmap_f = None

        self._fhook = self.target_module.register_forward_hook(self._forward_hook)
        self._bhook = self.target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, outputs):
        # stage4_fusion returns (s4_post, f4, ...)
        s4_post, f4, *_ = outputs
        self._fmap_s = s4_post.detach()
        self._fmap_f = f4.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        grad_s, grad_f, *_ = grad_out
        self._gmap_s = grad_s.detach()
        self._gmap_f = grad_f.detach()

    @torch.no_grad()
    def _normalize(self, t):
        t = t - t.min()
        if t.max() > 0:
            t = t / t.max()
        return t

    def _make_cam(self, fmap, gmap, up_size):
        # fmap,gmap: [B,C,H,W]
        w = gmap.mean(dim=(2, 3), keepdim=True)           # [B,C,1,1]
        cam = (w * fmap).sum(dim=1, keepdim=True)         # [B,1,h,w]
        cam = F.relu(cam)
        cams = []
        for b in range(cam.size(0)):
            c = self._normalize(cam[b:b+1])               # [1,1,h,w] in 0..1
            c = F.interpolate(c, size=up_size, mode="bilinear", align_corners=False)
            c = self._normalize(c)
            cams.append(c)
        return torch.cat(cams, dim=0)                     # [B,1,H,W]

    def __call__(self, x, target_logit_idx=None, combine="avg"):
        """
        x: [B,3,H,W] preprocessed tensor
        For BSFNet (binary), we backprop the single logit.
        Returns dict: {'spatial','frequency', 'combined'(optional)} each [B,1,H,W] in [0,1].
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)

        if logits.shape[1] == 1:
            scalar = logits[:, 0].sum()
        else:
            if target_logit_idx is None:
                target_logit_idx = int(torch.argmax(logits, dim=1)[0].item())
            scalar = logits[:, target_logit_idx].sum()

        scalar.backward(retain_graph=False)

        H, W = x.shape[-2:]
        cam_s = self._make_cam(self._fmap_s, self._gmap_s, (H, W))
        cam_f = self._make_cam(self._fmap_f, self._gmap_f, (H, W))

        out = {"spatial": cam_s.cpu(), "frequency": cam_f.cpu()}
        if combine is not None:
            if combine == "avg":
                comb = 0.5 * (cam_s + cam_f)
            elif combine == "max":
                comb = torch.maximum(cam_s, cam_f)
            else:
                raise ValueError("combine must be 'avg', 'max' or None")
            out["combined"] = self._normalize(comb).cpu()
        return out

    def close(self):
        self._fhook.remove()
        self._bhook.remove()


# ---------- Visualization helpers ----------

def overlay_heatmap_bgr(img_bgr: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    img_bgr: HxWx3 uint8 (OpenCV BGR)
    heatmap01: HxW float in [0,1]
    Returns: HxWx3 uint8 (BGR)
    """
    hm_u8 = np.clip(heatmap01 * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(color, alpha, img_bgr, 1 - alpha, 0)
    return out


def bsfnet_gradcam_png(
    model,
    pil_image: Image.Image,
    transform, 
    device,
    combine: str = "avg",
    alpha: float = 0.45,
    return_mode: str = "triptych" # 'triptych' | 'combined' | 'spatial' | 'frequency'
) -> bytes:
    """
    Runs Grad-CAM on a PIL image and returns a PNG (bytes).
    - triptych: spatial | frequency | combined (stitched)
    - otherwise returns a single overlay
    """

    x = transform(pil_image).unsqueeze(0).to(device)

    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    cam = GradCAM_BSFNet(model, device=device)
    cams = cam(x, combine=combine)
    cam.close()

    spatial = cv2.resize(cams["spatial"][0, 0].numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
    freq    = cv2.resize(cams["frequency"][0, 0].numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
    comb    = cv2.resize(cams["combined"][0, 0].numpy(), (W, H), interpolation=cv2.INTER_LINEAR)

    ov_s = overlay_heatmap_bgr(img_bgr, spatial, alpha=alpha)
    ov_f = overlay_heatmap_bgr(img_bgr, freq,    alpha=alpha)
    ov_c = overlay_heatmap_bgr(img_bgr, comb,    alpha=alpha)

    if return_mode == "triptych":
        canvas = np.concatenate([ov_s, ov_f, ov_c], axis=1)
    elif return_mode == "combined":
        canvas = ov_c
    elif return_mode == "spatial":
        canvas = ov_s
    elif return_mode == "frequency":
        canvas = ov_f
    else:
        raise ValueError("return_mode must be one of: 'triptych','combined','spatial','frequency'")

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
