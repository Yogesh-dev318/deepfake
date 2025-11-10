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

# from utils.paths import IMAGE_WEIGHTS, AUDIO_WEIGHTS, require_file


# # image_model_path = "/Users/yogeshpoonia/Documents/100xdev/deepfake/backend/weights/image_weights/bsfnet_deepfake.pth"
# # audio_model_path = "/Users/yogeshpoonia/Documents/100xdev/deepfake/backend/weights/audio_weights/best_dbmfc_weights_only.pth"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image_state = torch.load(require_file(IMAGE_WEIGHTS,
#     "Expected at backend/models/weights/image_weights/bsfnet_deepfake.pth"
# ), map_location=device)

# audio_state = torch.load(require_file(AUDIO_WEIGHTS,
#     "Expected at backend/models/weights/audio_weights/best_dbmfc_weights_only.pth"
# ), map_location=device)


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
#     except Exception:
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
#         return jsonify({"message": "No file uploaded"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"message": "Empty filename"}), 400

#     try:
#         image = Image.open(file.stream).convert("RGB")
#         img_tensor = transform(image).unsqueeze(0)  # (1,3,128,128)

#         with torch.no_grad():
#             output = image_model(img_tensor)
#             prob = torch.sigmoid(output).item()
#             prediction = int(prob > 0.5)

#         return jsonify({
#             "prediction": "fake" if prediction == 1 else "real",
#             "confidence": round(prob, 4),
#             "label": "Fake" if prediction else "Real"
#         })
#     except Exception as e:
#         return jsonify({"message": str(e)}), 500

# # -------------------------
# # Audio prediction endpoint
# # -------------------------
# @app.route("/predict_audio", methods=["POST"])
# def predict_audio():
#     """
#     Accepts an uploaded audio file (e.g., .flac, .wav, .mp3, .ogg). Field name: 'file'.
#     Returns JSON compatible with your frontend:
#     {
#       "prediction": "fake" or "real",
#       "confidence": 0.8723,   # probability of 'fake' (spoof) class
#       "logits": [...],
#       "probs": [...]
#     }
#     """
#     if "file" not in request.files:
#         return jsonify({"message": "No file uploaded"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"message": "Empty filename"}), 400

#     tmp_path = None
#     try:
#         # Save uploaded file to a temporary file so torchaudio / ASVSpoofDataset can read it
#         suffix = os.path.splitext(file.filename)[1] or ".wav"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             tmp_path = tmp.name
#             file.save(tmp_path)

#         # If torchaudio backend needs to be set, uncomment & adjust:
#         # torchaudio.set_audio_backend("sox_io")

#         # Preprocess using your ASVSpoofDataset (augment=False for inference)
#         ds = ASVSpoofDataset([tmp_path], [0], augment=False)
#         sample = ds[0]
#         mel, spec = sample["mel"], sample["spec"]

#         # Batchify & send to device
#         mel_b = mel.unsqueeze(0).to(device)   # (1, F, T)
#         spec_b = spec.unsqueeze(0).to(device) # (1, F, T)

#         # Forward pass
#         try:
#             (logits, emb, _), used_shape = try_forward(audio_model, mel_b, spec_b)
#         except RuntimeError as e:
#             return jsonify({"message": f"Model forward failed: {e}"}), 500

#         # Convert to probabilities
#         probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]  # shape (2,)
#         pred_idx = int(probs.argmax())  # 0 = bonafide (real), 1 = spoof (fake)
#         fake_prob = float(probs[1])     # probability for spoof/fake class

#         return jsonify({
#             "prediction": "fake" if pred_idx == 1 else "real",
#             "confidence": round(fake_prob, 4),
#             "logits": logits.detach().cpu().numpy()[0].tolist(),
#             "probs": probs.tolist()
#         })
#     except Exception as e:
#         return jsonify({"message": str(e)}), 500
#     finally:
#         try:
#             if tmp_path and os.path.exists(tmp_path):
#                 os.unlink(tmp_path)
#         except Exception:
#             pass

# if __name__ == "__main__":
#     app.run(debug=True)



# backend/app.py
import os
import tempfile
from pathlib import Path
import io
import uuid
from flask import send_file

import torch
import torch.nn as nn
import torchaudio
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS


from preprocessing.audio_process import CFG, ASVSpoofDataset
from models.imageModel.BSFNet import BSFNet
from models.audioModel.DBMFC import DBMFC

from utils.paths import IMAGE_WEIGHTS, AUDIO_WEIGHTS, require_file
from features.localization.image_saliency import bsfnet_gradcam_png
from features.localization.audio_timeline import audio_saliency_png_from_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UPLOAD_CACHE_DIR = os.path.join(tempfile.gettempdir(), "df_img_cache")
os.makedirs(UPLOAD_CACHE_DIR, exist_ok=True)

AUDIO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "df_audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

def load_state_dict_safely(model: nn.Module, ckpt_path: Path, device: torch.device):
    ckpt_path = require_file(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    try:
        model.load_state_dict(ckpt, strict=True)
    except Exception:
        model.load_state_dict(ckpt, strict=False)


def try_forward(model, mel_b, spec_b):
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


image_model = BSFNet(num_classes=1).to(device)
load_state_dict_safely(
    image_model,
    require_file(
        IMAGE_WEIGHTS,
        "Expected at backend/models/weights/image_weights/bsfnet_deepfake.pth"
    ),
    device=device,
)
image_model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


audio_model = DBMFC().to(device)
try:
    load_state_dict_safely(
        audio_model,
        require_file(
            AUDIO_WEIGHTS,
            "Expected at backend/models/weights/audio_weights/best_dbmfc_weights_only.pth"
        ),
        device=device,
    )
    audio_model.eval()
except Exception as e:
    print(f"[WARN] Audio model failed to load: {e}")
    audio_model.eval()


app = Flask(__name__)
CORS(app)


# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"message": "No file uploaded"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"message": "Empty filename"}), 400

#     try:
#         image = Image.open(file.stream).convert("RGB")
#         img_tensor = transform(image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             output = image_model(img_tensor)
#             if output.ndim > 1:
#                 output = output.view(-1)
#             prob = torch.sigmoid(output)[0].item()
#             prediction = int(prob > 0.5)

#         return jsonify({
#             "prediction": "fake" if prediction == 1 else "real",
#             "confidence": round(prob, 4),
#             "label": "Fake" if prediction else "Real"
#         })
#     except Exception as e:
#         return jsonify({"message": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "Empty filename"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")

        # NEW: persist the exact original image for Grad-CAM reuse
        image_id = str(uuid.uuid4())
        cache_path = os.path.join(UPLOAD_CACHE_DIR, f"{image_id}.png")
        image.save(cache_path)

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = image_model(img_tensor)
            if output.ndim > 1:
                output = output.view(-1)
            prob = torch.sigmoid(output)[0].item()
            prediction = int(prob > 0.5)

        return jsonify({
            "prediction": "fake" if prediction == 1 else "real",
            "confidence": round(prob, 4),
            "label": "Fake" if prediction else "Real",
            "image_id": image_id   # NEW: give frontend a handle to reuse the same image
        })
    except Exception as e:
        return jsonify({"message": str(e)}), 500



# @app.route("/predict_audio", methods=["POST"])
# def predict_audio():
#     """
#     Accepts uploaded audio (flac/wav/mp3/ogg). Field name: 'file'.
#     Returns:
#       {
#         "prediction": "fake" | "real",
#         "confidence": P(spoof),
#         "logits": [...],
#         "probs": [...]
#       }
#     """
#     if "file" not in request.files:
#         return jsonify({"message": "No file uploaded"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"message": "Empty filename"}), 400

#     tmp_path = None
#     try:
#         suffix = os.path.splitext(file.filename)[1] or ".wav"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             tmp_path = tmp.name
#             file.save(tmp_path)

#         ds = ASVSpoofDataset([tmp_path], [0], augment=False)
#         sample = ds[0]
#         mel, spec = sample["mel"], sample["spec"]

#         mel_b = mel.unsqueeze(0).to(device)   # (1, F, T)
#         spec_b = spec.unsqueeze(0).to(device) # (1, F, T)

#         try:
#             (logits, emb, _), used_shape = try_forward(audio_model, mel_b, spec_b)
#         except RuntimeError as e:
#             return jsonify({"message": f"Model forward failed: {e}"}), 500

#         probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]  # (2,)
#         pred_idx = int(probs.argmax())  # 0=real(bonafide), 1=fake(spoof)
#         fake_prob = float(probs[1])

#         return jsonify({
#             "prediction": "fake" if pred_idx == 1 else "real",
#             "confidence": round(fake_prob, 4),
#             "logits": logits.detach().cpu().numpy()[0].tolist(),
#             "probs": probs.tolist()
#         })
#     except Exception as e:
#         return jsonify({"message": str(e)}), 500
#     finally:
#         try:
#             if tmp_path and os.path.exists(tmp_path):
#                 os.unlink(tmp_path)
#         except Exception:
#             pass

@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "Empty filename"}), 400

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        # NEW: persist a copy to cache (so we can reuse on /audio_saliency)
        audio_id = str(uuid.uuid4())
        cache_path = os.path.join(AUDIO_CACHE_DIR, f"{audio_id}{suffix}")
        try:
            # copy to cache
            with open(tmp_path, "rb") as src, open(cache_path, "wb") as dst:
                dst.write(src.read())
        except Exception as e:
            return jsonify({"message": f"Failed to cache audio: {e}"}), 500

        ds = ASVSpoofDataset([cache_path], [0], augment=False)  # use cached path
        sample = ds[0]
        mel, spec = sample["mel"], sample["spec"]

        mel_b = mel.unsqueeze(0).to(device)   # (1, F, T)
        spec_b = spec.unsqueeze(0).to(device) # (1, F, T)

        try:
            (logits, emb, _), used_shape = try_forward(audio_model, mel_b, spec_b)
        except RuntimeError as e:
            return jsonify({"message": f"Model forward failed: {e}"}), 500

        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]  # (2,)
        pred_idx = int(probs.argmax())  # 0=real, 1=fake
        fake_prob = float(probs[1])

        return jsonify({
            "prediction": "fake" if pred_idx == 1 else "real",
            "confidence": round(fake_prob, 4),
            "logits": logits.detach().cpu().numpy()[0].tolist(),
            "probs": probs.tolist(),
            "audio_id": audio_id   # NEW: return handle to reuse same audio
        })
    except Exception as e:
        return jsonify({"message": str(e)}), 500
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)  # keep only cached copy
        except Exception:
            pass



@app.route("/gradcam", methods=["POST"])
def gradcam_image():
    """
    Usage:
      - Preferred: JSON { "image_id": "<uuid-from-/predict>" }
      - Fallback: multipart/form-data with field 'file' (image)
    Returns:
      PNG image (heatmap overlay). By default: spatial | frequency | combined.
    """
    try:
        pil = None

        # Preferred path: reuse cached image by ID
        if request.is_json and "image_id" in request.json:
            image_id = str(request.json["image_id"])
            cache_path = os.path.join(UPLOAD_CACHE_DIR, f"{image_id}.png")
            if not (os.path.exists(cache_path) and os.path.isfile(cache_path)):
                return jsonify({"message": "Invalid or expired image_id"}), 400
            pil = Image.open(cache_path).convert("RGB")

        # Fallback: accept a new file
        elif "file" in request.files:
            f = request.files["file"]
            if f.filename == "":
                return jsonify({"message": "Empty filename"}), 400
            pil = Image.open(f.stream).convert("RGB")

        else:
            return jsonify({"message": "Provide JSON {'image_id': ...} or upload 'file'"}), 400

        png_bytes = bsfnet_gradcam_png(
            model=image_model,
            pil_image=pil,
            transform=transform,
            device=device,
            combine="avg",
            alpha=0.45,
            return_mode="triptych"  # 'combined' | 'spatial' | 'frequency' also supported
        )

        return send_file(
            io.BytesIO(png_bytes),
            mimetype="image/png",
            as_attachment=False,
            download_name="gradcam.png",
            etag=False
        )
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/audio_saliency", methods=["POST"])
def audio_saliency():
    """
    Usage: JSON { "audio_id": "<uuid returned by /predict_audio>" }
    Returns: PNG image (triptych by default).
    """
    try:
        if not request.is_json or "audio_id" not in request.json:
            return jsonify({"message": "Provide JSON with 'audio_id'"}), 400

        audio_id = str(request.json["audio_id"])
        # find file with any typical audio extension
        candidates = [f for f in os.listdir(AUDIO_CACHE_DIR) if f.startswith(audio_id)]
        if not candidates:
            return jsonify({"message": "Invalid or expired audio_id"}), 400
        cache_path = os.path.join(AUDIO_CACHE_DIR, candidates[0])

        png_bytes = audio_saliency_png_from_path(
            model=audio_model,
            audio_path=cache_path,
            device=device,
            alpha=0.45,
            return_mode="triptych"  # 'mel' | 'spec' | 'combined' also available
        )

        return send_file(
            io.BytesIO(png_bytes),
            mimetype="image/png",
            as_attachment=False,
            download_name="audio_saliency.png",
            etag=False
        )
    except Exception as e:
        return jsonify({"message": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
