# import os
# import tempfile
# import time
# import torch
# import torch.nn as nn
# import torchaudio
# from torchvision import transforms
# from PIL import Image
# from flask import Flask, request, jsonify
# from flask_cors import CORS  # allow Next.js to call API

# # project modules / preprocessing / models
# from preprocessing.audio_process import CFG, ASVSpoofDataset
# from models.imageModel.BSFNet import BSFNet
# from models.audioModel.DBMFC import DBMFC

# # -------------------------
# # Paths to weights
# # -------------------------
# image_model_path = "/Users/yogeshpoonia/Documents/100xdev/deepfake/backend/weights/image_weights/bsfnet_deepfake.pth"
# audio_model_path = "/Users/yogeshpoonia/Documents/100xdev/deepfake/backend/weights/audio_weights/best_dbmfc_weights_only.pth"

# # -------------------------
# # Device
# # -------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -------------------------
# # Helper: robust state dict loader
# # -------------------------
# def load_state_dict_safely(model, ckpt_path, device):
#     """
#     Loads weights robustly:
#     - supports checkpoints saved as full dict {'state_dict': ..., ...}
#     - removes 'module.' prefixes if saved via DataParallel
#     - tries strict first, then non-strict
#     """
#     ckpt = torch.load(ckpt_path, map_location=device)

#     # if ckpt is a dict with nested state, try common keys
#     if isinstance(ckpt, dict):
#         for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
#             if key in ckpt and isinstance(ckpt[key], dict):
#                 ckpt = ckpt[key]
#                 break

#     # remove 'module.' prefix if present
#     if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt.keys()):
#         ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

#     # Attempt strict load, else non-strict
#     try:
#         model.load_state_dict(ckpt, strict=True)
#         # no return necessary
#     except Exception as e:
#         # try non-strict
#         model.load_state_dict(ckpt, strict=False)

# # -------------------------
# # Helper: try forward with flexible dims
# # -------------------------
# def try_forward(model, mel_b, spec_b):
#     """
#     Try forward with shapes (B, F, T) — if that fails, try (B,1,F,T).
#     Returns (model_output, used_shape_description)
#     """
#     model.eval()
#     with torch.no_grad():
#         try:
#             out = model(mel_b, spec_b, mask_ratio=0.0, do_reconstruct=False)
#             return out, "3D (B,F,T)"
#         except Exception as e1:
#             try:
#                 out = model(mel_b.unsqueeze(1), spec_b.unsqueeze(1), mask_ratio=0.0, do_reconstruct=False)
#                 return out, "4D (B,1,F,T)"
#             except Exception as e2:
#                 raise RuntimeError(f"Forward failed.\n3D error: {e1}\n4D error: {e2}")

# # -------------------------
# # Load Image Model (same as before)
# # -------------------------
# image_model = BSFNet(num_classes=1)
# image_state = torch.load(image_model_path, map_location=torch.device("cpu"))
# if any(key.startswith("module.") for key in image_state.keys()):
#     image_state = {k.replace("module.", ""): v for k, v in image_state.items()}
# image_model.load_state_dict(image_state)
# image_model.eval()

# # -------------------------
# # Image transforms (same)
# # -------------------------
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# # -------------------------
# # Load Audio Model (DBMFC)
# # -------------------------
# audio_model = DBMFC().to(device)
# try:
#     load_state_dict_safely(audio_model, audio_model_path, device)
# except Exception as e:
#     # If loading fails, print but keep running so image API still works.
#     print(f"[WARN] Audio model failed to load: {e}")

# audio_model.eval()

# # -------------------------
# # Flask App
# # -------------------------
# app = Flask(__name__)
# CORS(app)  # enable CORS for frontend

# # -------------------------
# # Image prediction endpoint (unchanged)
# # -------------------------
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400

#     try:
#         image = Image.open(file.stream).convert("RGB")
#         img_tensor = transform(image).unsqueeze(0)  # (1,3,128,128)

#         with torch.no_grad():
#             output = image_model(img_tensor)
#             prob = torch.sigmoid(output).item()
#             prediction = int(prob > 0.5)

#         return jsonify({
#             "prediction": prediction,
#             "probability": round(prob, 4),
#             "label": "Fake" if prediction else "Real"
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # -------------------------
# # Audio prediction endpoint
# # -------------------------
# @app.route("/predict_audio", methods=["POST"])
# def predict_audio():
#     """
#     Accepts an uploaded audio file (e.g., .flac, .wav). Field name: 'file'.
#     Returns JSON:
#     {
#       "prediction": 1,                 # 0 = bonafide, 1 = spoof
#       "probability": 0.87,             # probability of 'spoof' class (index 1)
#       "label": "spoof",
#       "logits": [...],
#       "probs": [...]
#     }
#     """
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400

#     # Save uploaded file to a temporary file so torchaudio / your ASVSpoofDataset can read it
#     try:
#         suffix = os.path.splitext(file.filename)[1] or ".wav"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             tmp_path = tmp.name
#             file.save(tmp_path)

#         # Optional: choose torchaudio backend if needed (not required here)
#         # torchaudio.set_audio_backend("sox_io")  # uncomment if necessary

#         # Use your dataset preprocessing (no augmentation for inference)
#         ds = ASVSpoofDataset([tmp_path], [0], augment=False)
#         sample = ds[0]
#         mel, spec = sample["mel"], sample["spec"]

#         # Ensure tensors are on the correct device and batched
#         mel_b = mel.unsqueeze(0).to(device)   # (1, F, T) or (1,1,F,T) depending on model
#         spec_b = spec.unsqueeze(0).to(device) # (1, F, T)

#         # Forward pass (model returns: logits, emb, maybe recon)
#         try:
#             (logits, emb, _), used_shape = try_forward(audio_model, mel_b, spec_b)
#         except RuntimeError as e:
#             # cleanup temp file
#             os.unlink(tmp_path)
#             return jsonify({"error": f"Model forward failed: {e}"}), 500

#         elapsed_ms = (time.time() - time.time()) * 1000  # placeholder if you want latency; left as 0

#         # Convert to probabilities
#         probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]  # shape (2,)
#         pred = int(probs.argmax())  # 0 = bonafide, 1 = spoof
#         spoof_prob = float(probs[1])  # probability for spoof class

#         # Cleanup tmp file
#         os.unlink(tmp_path)

#         return jsonify({
#             "prediction": pred,
#             "probability": round(spoof_prob, 4),
#             "label": "spoof" if pred == 1 else "bonafide",
#             "logits": logits.detach().cpu().numpy()[0].tolist(),
#             "probs": probs.tolist()
#         })
#     except Exception as e:
#         # Try to cleanup tmp file if it exists
#         try:
#             if 'tmp_path' in locals() and os.path.exists(tmp_path):
#                 os.unlink(tmp_path)
#         except Exception:
#             pass
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)

import os
import tempfile
import time
import torch
import torch.nn as nn
import torchaudio
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS  # allow Next.js to call API

# project modules / preprocessing / models
from preprocessing.audio_process import CFG, ASVSpoofDataset
from models.imageModel.BSFNet import BSFNet
from models.audioModel.DBMFC import DBMFC

# -------------------------
# Paths to weights
# -------------------------
image_model_path = "/Users/yogeshpoonia/Documents/100xdev/deepfake/backend/weights/image_weights/bsfnet_deepfake.pth"
audio_model_path = "/Users/yogeshpoonia/Documents/100xdev/deepfake/backend/weights/audio_weights/best_dbmfc_weights_only.pth"
# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Helper: robust state dict loader
# -------------------------
def load_state_dict_safely(model, ckpt_path, device):
    """
    Loads weights robustly:
    - supports checkpoints saved as full dict {'state_dict': ..., ...}
    - removes 'module.' prefixes if saved via DataParallel
    - tries strict first, then non-strict
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # if ckpt is a dict with nested state, try common keys
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    # remove 'module.' prefix if present
    if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    # Attempt strict load, else non-strict
    try:
        model.load_state_dict(ckpt, strict=True)
    except Exception:
        model.load_state_dict(ckpt, strict=False)

# -------------------------
# Helper: try forward with flexible dims
# -------------------------
def try_forward(model, mel_b, spec_b):
    """
    Try forward with shapes (B, F, T) — if that fails, try (B,1,F,T).
    Returns (model_output, used_shape_description)
    """
    model.eval()
    with torch.no_grad():
        try:
            out = model(mel_b, spec_b, mask_ratio=0.0, do_reconstruct=False)
            return out, "3D (B,F,T)"
        except Exception as e1:
            try:
                out = model(mel_b.unsqueeze(1), spec_b.unsqueeze(1), mask_ratio=0.0, do_reconstruct=False)
                return out, "4D (B,1,F,T)"
            except Exception as e2:
                raise RuntimeError(f"Forward failed.\n3D error: {e1}\n4D error: {e2}")

# -------------------------
# Load Image Model (same as before)
# -------------------------
image_model = BSFNet(num_classes=1)
image_state = torch.load(image_model_path, map_location=torch.device("cpu"))
if any(key.startswith("module.") for key in image_state.keys()):
    image_state = {k.replace("module.", ""): v for k, v in image_state.items()}
image_model.load_state_dict(image_state)
image_model.eval()

# -------------------------
# Image transforms (same)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -------------------------
# Load Audio Model (DBMFC)
# -------------------------
audio_model = DBMFC().to(device)
try:
    load_state_dict_safely(audio_model, audio_model_path, device)
except Exception as e:
    # If loading fails, print but keep running so image API still works.
    print(f"[WARN] Audio model failed to load: {e}")

audio_model.eval()

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)
CORS(app)  # enable CORS for frontend

# -------------------------
# Image prediction endpoint (unchanged)
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "Empty filename"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)  # (1,3,128,128)

        with torch.no_grad():
            output = image_model(img_tensor)
            prob = torch.sigmoid(output).item()
            prediction = int(prob > 0.5)

        return jsonify({
            "prediction": "fake" if prediction == 1 else "real",
            "confidence": round(prob, 4),
            "label": "Fake" if prediction else "Real"
        })
    except Exception as e:
        return jsonify({"message": str(e)}), 500

# -------------------------
# Audio prediction endpoint
# -------------------------
@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    """
    Accepts an uploaded audio file (e.g., .flac, .wav, .mp3, .ogg). Field name: 'file'.
    Returns JSON compatible with your frontend:
    {
      "prediction": "fake" or "real",
      "confidence": 0.8723,   # probability of 'fake' (spoof) class
      "logits": [...],
      "probs": [...]
    }
    """
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "Empty filename"}), 400

    tmp_path = None
    try:
        # Save uploaded file to a temporary file so torchaudio / ASVSpoofDataset can read it
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        # If torchaudio backend needs to be set, uncomment & adjust:
        # torchaudio.set_audio_backend("sox_io")

        # Preprocess using your ASVSpoofDataset (augment=False for inference)
        ds = ASVSpoofDataset([tmp_path], [0], augment=False)
        sample = ds[0]
        mel, spec = sample["mel"], sample["spec"]

        # Batchify & send to device
        mel_b = mel.unsqueeze(0).to(device)   # (1, F, T)
        spec_b = spec.unsqueeze(0).to(device) # (1, F, T)

        # Forward pass
        try:
            (logits, emb, _), used_shape = try_forward(audio_model, mel_b, spec_b)
        except RuntimeError as e:
            return jsonify({"message": f"Model forward failed: {e}"}), 500

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]  # shape (2,)
        pred_idx = int(probs.argmax())  # 0 = bonafide (real), 1 = spoof (fake)
        fake_prob = float(probs[1])     # probability for spoof/fake class

        return jsonify({
            "prediction": "fake" if pred_idx == 1 else "real",
            "confidence": round(fake_prob, 4),
            "logits": logits.detach().cpu().numpy()[0].tolist(),
            "probs": probs.tolist()
        })
    except Exception as e:
        return jsonify({"message": str(e)}), 500
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(debug=True)
