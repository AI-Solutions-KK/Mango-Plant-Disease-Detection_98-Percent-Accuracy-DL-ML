# ================================
# FILE: app/inference.py
# ================================

import torch
import timm
import numpy as np
from PIL import Image
import pickle

from app.config import CLF_FILE, IMAGE_SIZE, MODEL_NAME

device = torch.device("cpu")

model = None
classifier = None
label_encoder = None
normalizer = None


# ---------------- LOAD MODELS ----------------
def load_models():
    global model, classifier, label_encoder, normalizer

    if model is not None:
        return

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0
    ).to(device).eval()

    with open(CLF_FILE, "rb") as f:
        obj = pickle.load(f)

    classifier = obj["clf"]
    label_encoder = obj["le"]
    normalizer = obj["norm"]

    print("✅ Inference models loaded")


# ---------------- PREPROCESS ----------------
def preprocess_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)

    img = np.array(img, dtype=np.float32) / 255.0  # 🔥 force float32

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))

    tensor = torch.from_numpy(img).unsqueeze(0).float()  # 🔥 force float32
    return tensor.to(device)


# ---------------- FEATURES ----------------
def extract_features(image_path: str):
    if model is None:
        load_models()

    img_tensor = preprocess_image(image_path)

    with torch.no_grad():
        features = model(img_tensor)

    return features.cpu().numpy().astype(np.float32).flatten()


# ---------------- PREDICT ----------------
def predict_image(image_path: str, top_k: int = 3):
    try:
        features = extract_features(image_path)

        features = normalizer.transform(features.reshape(1, -1))

        pred_idx = classifier.predict(features)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        probs = classifier.predict_proba(features)[0]
        confidence = float(probs[pred_idx])

        return {
            "predicted_label": pred_label,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}
