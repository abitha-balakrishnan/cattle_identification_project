from flask import Flask, render_template, request, session
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename
from datetime import datetime

# =====================================
# Flask App Setup
# =====================================
app = Flask(__name__)
app.secret_key = "cattleai_india_2024_secret"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =====================================
# Class Names
# =====================================
class_names = [
    "Ayrshire cattle",
    "Brown Swiss cattle",
    "Holstein Friesian cattle",
    "Jersey cattle",
    "Red Dane cattle"
]

# =====================================
# Enhanced Breed Information
# =====================================
breed_info = {
    "Ayrshire cattle": {
        "description": "A high milk-producing dairy breed known for strong adaptability, hardiness, and suitability across diverse climates including parts of India.",
        "origin": "Scotland, UK",
        "milk_yield": "5,000 – 8,000 kg/year",
        "purpose": "Dairy",
        "fat_content": "3.9%",
        "characteristics": "Medium-sized, red and white patches, strong feet and legs, well-attached udder",
        "icon": "🐄"
    },
    "Brown Swiss cattle": {
        "description": "Famous for exceptional longevity, strength, and high-quality milk with elevated protein content. Adapts well to tropical environments.",
        "origin": "Switzerland",
        "milk_yield": "7,000 – 10,000 kg/year",
        "purpose": "Dual Purpose",
        "fat_content": "4.0%",
        "characteristics": "Brown coat, large frame, calm temperament, long productive life",
        "icon": "🐄"
    },
    "Holstein Friesian cattle": {
        "description": "World's highest milk-producing dairy cattle, easily recognized by iconic black and white markings. Widely used in Indian crossbreeding programs.",
        "origin": "Netherlands",
        "milk_yield": "9,000 – 12,000 kg/year",
        "purpose": "Dairy",
        "fat_content": "3.7%",
        "characteristics": "Distinctive black-white patches, large body, high fertility, strong milk veins",
        "icon": "🐄"
    },
    "Jersey cattle": {
        "description": "Produces milk exceptionally rich in butterfat and protein. Highly heat-tolerant and popular with Indian small dairy farmers.",
        "origin": "Jersey Island, UK",
        "milk_yield": "4,500 – 7,000 kg/year",
        "purpose": "Dairy",
        "fat_content": "5.2%",
        "characteristics": "Fawn or brown color, small compact frame, large expressive eyes, high feed efficiency",
        "icon": "🐄"
    },
    "Red Dane cattle": {
        "description": "Hardy dual-purpose breed equally suited for milk and meat production, with excellent disease resistance and adaptability.",
        "origin": "Denmark",
        "milk_yield": "6,000 – 9,000 kg/year",
        "purpose": "Dual Purpose",
        "fat_content": "4.2%",
        "characteristics": "Uniform red coat, strong muscular build, great disease resistance, easy calving",
        "icon": "🐄"
    }
}

# =====================================
# Device & Model
# =====================================
device = torch.device("cpu")

model = models.resnet18(weights=None)
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("cattle_model.pth", map_location=device))
model.to(device)
model.eval()

print("✅ Model Loaded Successfully")

# =====================================
# Image Transform
# =====================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================
# Home Route
# =====================================
@app.route("/", methods=["GET", "POST"])
def index():

    prediction    = None
    confidence    = None
    filename      = None
    description   = None
    top_predictions = None
    timestamp     = None

    if "history" not in session:
        session["history"] = []

    if request.method == "POST":

        if "file" not in request.files:
            return render_template("index.html", history=session.get("history", []))

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", history=session.get("history", []))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # ==========================
        # Image Processing
        # ==========================
        image = Image.open(filepath).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        # ==========================
        # Prediction
        # ==========================
        with torch.no_grad():
            outputs       = model(tensor)
            probabilities = F.softmax(outputs, dim=1)

            # Top-3 Predictions
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            top_predictions = [
                {
                    "name":       class_names[idx.item()],
                    "confidence": round(prob.item() * 100, 2)
                }
                for prob, idx in zip(top3_prob[0], top3_idx[0])
            ]

            confidence_val, predicted = torch.max(probabilities, 1)
            prediction  = class_names[predicted.item()]
            confidence  = round(confidence_val.item() * 100, 2)
            description = breed_info.get(prediction)
            timestamp   = datetime.now().strftime("%d %b %Y, %I:%M %p")

            # Session History (last 5)
            history = session.get("history", [])
            history.insert(0, {
                "prediction": prediction,
                "confidence": confidence,
                "filename":   "uploads/" + filename,
                "timestamp":  timestamp
            })
            session["history"] = history[:5]
            session.modified = True

    return render_template(
        "index.html",
        prediction      = prediction,
        confidence      = confidence,
        filename        = ("uploads/" + filename) if filename else None,
        description     = description,
        top_predictions = top_predictions,
        history         = session.get("history", [])
    )

# =====================================
# Clear History Route
# =====================================
@app.route("/clear-history")
def clear_history():
    session["history"] = []
    session.modified = True
    return "cleared"

# =====================================
# Run Server
# =====================================
if __name__ == "__main__":
    app.run(debug=True)
