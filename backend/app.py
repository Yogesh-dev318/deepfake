import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS  # allow Next.js to call API

from models.BSFNet import BSFNet

# -------------------------
# Load Model
# -------------------------
model_path = "E:\DeepFake Demo Project - Capstone\Main App\deepfake\backend\weights\bsfnet_deepfake.pth"

model = BSFNet(num_classes=1)
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# -------------------------
# Define Transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)
CORS(app)  # enable CORS for frontend

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            prediction = int(prob > 0.5)

        return jsonify({
            "prediction": prediction,
            "probability": round(prob, 4),
            "label": "Fake" if prediction else "Real"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
