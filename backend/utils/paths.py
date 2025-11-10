from pathlib import Path
import os

BACKEND_DIR = Path(__file__).resolve().parent.parent

WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", BACKEND_DIR / "weights"))

IMAGE_WEIGHTS = Path(os.getenv(
    "IMAGE_WEIGHTS",
    WEIGHTS_DIR / "image_weights" / "bsfnet_deepfake.pth"
))
AUDIO_WEIGHTS = Path(os.getenv(
    "AUDIO_WEIGHTS",
    WEIGHTS_DIR / "audio_weights" / "best_dbmfc_weights_only.pth"
))

def require_file(path: Path, hint: str = "") -> Path:
    path = Path(path).resolve()
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\nHint: {hint}"
        msg += (
            "\nYou can override via env vars: WEIGHTS_DIR / IMAGE_WEIGHTS / AUDIO_WEIGHTS."
            f"\nResolved BACKEND_DIR = {BACKEND_DIR}"
        )
        raise FileNotFoundError(msg)
    return path
