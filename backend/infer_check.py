import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchaudio

# --- project root so imports work ---
PROJECT_ROOT = r"E:\DeepFake Demo Project - Capstone\Main App\deepfake\backend"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# your modules
from preprocessing.audio_process import CFG, ASVSpoofDataset
from models.audioModel.artifact_branch import ArtifactsBranch
from models.audioModel.structural_branch import StructuralBranch

# -------------------------
# DBM-FC (same as your code)
# -------------------------
class DBMFC(nn.Module):
    def __init__(self, structural_module=None, artifacts_module=None, 
                 embedding_dim=CFG["embedding_dim"], num_classes=2):
        super().__init__()
        self.structural = structural_module if structural_module is not None else StructuralBranch()
        self.artifacts = artifacts_module if artifacts_module is not None else ArtifactsBranch()
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim*2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim*2, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, mel, spec, mask_ratio=0.0, do_reconstruct=False):
        if do_reconstruct:
            s_emb, recon = self.structural(mel, mask_ratio=mask_ratio, do_reconstruct=True)
        else:
            s_emb = self.structural(mel, mask_ratio=mask_ratio, do_reconstruct=False)
            recon = None
        a_emb = self.artifacts(spec)
        fused = torch.cat([s_emb, a_emb], dim=1)
        emb = self.fusion(fused)
        logits = self.classifier(emb)
        return logits, emb, recon

def load_state_dict_safely(model, ckpt_path, device):
    """
    Loads weights robustly:
    - supports checkpoints saved as full dict {'model_state_dict': ..., ...}
    - removes 'module.' prefixes if saved via DataParallel
    - tries strict first, then non-strict
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        # candidate keys
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    try:
        missing, unexpected = model.load_state_dict(ckpt, strict=True)
        print("Loaded weights (strict=True).")
    except Exception as e:
        print(f"Strict load failed: {e}\nTrying non-strict...")
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print("Loaded weights (strict=False).")
    if missing:
        print(f"- Missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"- Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")

def try_forward(model, mel_b, spec_b):
    """
    Try (B,F,T); if it fails (e.g., Conv2d expects channel), try (B,1,F,T).
    Returns (outputs, shape_used)
    """
    model.eval()
    with torch.no_grad():
        try:
            return model(mel_b, spec_b, mask_ratio=0.0, do_reconstruct=False), "3D (B,F,T)"
        except Exception as e1:
            try:
                return model(mel_b.unsqueeze(1), spec_b.unsqueeze(1), mask_ratio=0.0, do_reconstruct=False), "4D (B,1,F,T)"
            except Exception as e2:
                raise RuntimeError(f"Forward failed.\n3D error: {e1}\n4D error: {e2}")

def main():
    parser = argparse.ArgumentParser(description="DBM-FC single-file inference sanity check")
    parser.add_argument("--flac", required=True, help="Path to a .flac file")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth/.pt)")
    parser.add_argument("--backend", default="", choices=["", "soundfile", "sox_io"],
                        help="Optional torchaudio backend override if FLAC load errors")
    args = parser.parse_args()

    if args.backend:
        torchaudio.set_audio_backend(args.backend)

    assert os.path.exists(args.flac), f"FLAC not found: {args.flac}"
    assert os.path.exists(args.weights), f"Weights not found: {args.weights}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # dataset preprocess (no augment) to keep it aligned with training pipeline
    ds = ASVSpoofDataset([args.flac], [0], augment=False)
    sample = ds[0]
    mel, spec = sample["mel"], sample["spec"]
    print(f"Preprocessed shapes → mel: {tuple(mel.shape)}, spec: {tuple(spec.shape)}")

    # batchify & move
    mel_b = mel.unsqueeze(0).to(device)   # (1,F,T)
    spec_b = spec.unsqueeze(0).to(device) # (1,F,T)

    # model
    model = DBMFC().to(device)
    load_state_dict_safely(model, args.weights, device)

    # forward
    t0 = time.time()
    (logits, emb, _), shape_used = try_forward(model, mel_b, spec_b)
    elapsed = (time.time() - t0) * 1000

    # outputs
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
    pred = int(probs.argmax())

    print(f"\n=== Inference OK ===")
    print(f"Input used: {shape_used}")
    print(f"Logits: {logits.detach().cpu().numpy()[0]}")
    print(f"Probs : {probs}  (index 0=bonafide, 1=spoof)")
    print(f"Pred  : {pred}  -> {'bonafide' if pred==0 else 'spoof'}")
    print(f"Embed shape: {tuple(emb.shape)}")
    print(f"Latency: {elapsed:.2f} ms\n")

if __name__ == "__main__":
    main()
