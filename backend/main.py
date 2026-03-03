"""
FastAPI Backend for Indian Cattle & Buffalo Breed Recognition System

This module implements the backend API for the cattle breed recognition system
using FastAPI, PyTorch, and supporting libraries.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os
import json
import uuid
from datetime import datetime
import base64
from pathlib import Path
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database.db_manager import get_db_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Indian Cattle & Buffalo Breed Recognition API",
    description="API for recognizing Indian cattle and buffalo breeds from images",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="../static"), name="static")

# Initialize database manager
db_manager = get_db_manager()

# Set up templates (for the web interface)
templates = Jinja2Templates(directory="../templates")

# Define upload directory
UPLOAD_DIR = Path("../static/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Class names for the model
CLASS_NAMES = [
    "Ayrshire cattle",
    "Brown Swiss cattle", 
    "Holstein Friesian cattle",
    "Jersey cattle",
    "Red Dane cattle"
]

# Updated to match the existing model checkpoint
# The model was trained with 5 classes, not 14
ORIGINAL_CLASS_NAMES = [
    "Gir",
    "Sahiwal", 
    "Ongole",
    "Murrah",
    "Jaffarabadi",
    "Red Sindhi",
    "Tharparkar",
    "Kankrej",
    "Hariana",
    "Rathi",
    "Surti",
    "Mehsana",
    "Nili_Ravi",
    "Pandharpuri"
]

# Enhanced breed information
BREED_INFO = {
    "Ayrshire cattle": {
        "name": "Ayrshire cattle",
        "type": "Cattle",
        "origin": "Scotland, UK",
        "description": "A high milk-producing dairy breed known for strong adaptability, hardiness, and suitability across diverse climates including parts of India.",
        "milk_yield": "5,000 – 8,000 kg/year",
        "fat_content": "3.9%",
        "characteristics": "Medium-sized, red and white patches, strong feet and legs, well-attached udder",
        "purpose": "Dairy",
        "icon": "🐄"
    },
    "Brown Swiss cattle": {
        "name": "Brown Swiss cattle",
        "type": "Cattle",
        "origin": "Switzerland",
        "description": "Famous for exceptional longevity, strength, and high-quality milk with elevated protein content. Adapts well to tropical environments.",
        "milk_yield": "7,000 – 10,000 kg/year",
        "fat_content": "4.0%",
        "characteristics": "Brown coat, large frame, calm temperament, long productive life",
        "purpose": "Dual Purpose",
        "icon": "🐄"
    },
    "Holstein Friesian cattle": {
        "name": "Holstein Friesian cattle",
        "type": "Cattle",
        "origin": "Netherlands",
        "description": "World's highest milk-producing dairy cattle, easily recognized by iconic black and white markings. Widely used in Indian crossbreeding programs.",
        "milk_yield": "9,000 – 12,000 kg/year",
        "fat_content": "3.7%",
        "characteristics": "Distinctive black-white patches, large body, high fertility, strong milk veins",
        "purpose": "Dairy",
        "icon": "🐄"
    },
    "Jersey cattle": {
        "name": "Jersey cattle",
        "type": "Cattle",
        "origin": "Jersey Island, UK",
        "description": "Produces milk exceptionally rich in butterfat and protein. Highly heat-tolerant and popular with Indian small dairy farmers.",
        "milk_yield": "4,500 – 7,000 kg/year",
        "fat_content": "5.2%",
        "characteristics": "Fawn or brown color, small compact frame, large expressive eyes, high feed efficiency",
        "purpose": "Dairy",
        "icon": "🐄"
    },
    "Red Dane cattle": {
        "name": "Red Dane cattle",
        "type": "Cattle",
        "origin": "Denmark",
        "description": "Hardy dual-purpose breed equally suited for milk and meat production, with excellent disease resistance and adaptability.",
        "milk_yield": "6,000 – 9,000 kg/year",
        "fat_content": "4.2%",
        "characteristics": "Uniform red coat, strong muscular build, great disease resistance, easy calving",
        "purpose": "Dual Purpose",
        "icon": "🐄"
    }
}

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
def load_model():
    """Load the trained model."""
    try:
        # Initialize model architecture
        model = models.resnet18(weights=None)
        num_classes = len(CLASS_NAMES)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load the trained weights
        model_path = "../cattle_model.pth"  # Adjust path as needed
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # Check if checkpoint is a state dict directly or wrapped in a dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
            logger.info("Model loaded successfully from checkpoint")
        else:
            logger.warning("Model checkpoint not found, using untrained model")
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

model = load_model()

# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    image: str  # Base64 encoded image

class PredictionResponse(BaseModel):
    breed: str
    confidence: float
    breed_info: Optional[Dict[str, Any]] = None
    top_predictions: List[Dict[str, Any]]

class PredictionHistoryItem(BaseModel):
    prediction: str
    confidence: float
    image_url: str
    timestamp: str

@app.get("/")
async def home(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict_breed(file: UploadFile = File(...)):
    """
    Predict the breed of cattle/buffalo from an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG) of cattle or buffalo
        
    Returns:
        PredictionResponse: Contains breed prediction, confidence, and breed information
    """
    try:
        # Validate file type
        content_type = file.content_type
        if content_type not in ["image/jpeg", "image/jpg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")
        
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Transform the image
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            top_predictions = []
            
            for prob, idx in zip(top3_probs[0], top3_indices[0]):
                class_idx = idx.item()
                if class_idx < len(CLASS_NAMES):
                    breed_name = CLASS_NAMES[class_idx]
                    confidence = round(prob.item() * 100, 2)
                    
                    top_predictions.append({
                        "breed": breed_name,
                        "confidence": confidence
                    })
            
            # Get the top prediction
            confidence_val, predicted_idx = torch.max(probabilities, 1)
            predicted_class_idx = predicted_idx.item()
            
            if predicted_class_idx >= len(CLASS_NAMES):
                raise HTTPException(status_code=500, detail="Model prediction index out of bounds")
            
            predicted_breed = CLASS_NAMES[predicted_class_idx]
            confidence = round(confidence_val.item() * 100, 2)
            
            # Get breed information
            breed_info = BREED_INFO.get(predicted_breed, {})
            
            # Save uploaded image
            file_extension = file.filename.split(".")[-1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
            file_path = UPLOAD_DIR / unique_filename
            
            with open(file_path, "wb") as f:
                f.write(contents)
            
            # Store prediction in database
            try:
                image_path_str = f"uploads/{unique_filename}"
                prediction_id = db_manager.insert_prediction(
                    image_name=file.filename,
                    image_path=image_path_str,
                    predicted_breed=predicted_breed,
                    confidence=confidence,
                    top_predictions=top_predictions,
                    user_ip=request.client.host if request.client else None
                )
                logger.info(f"Prediction stored in database with ID: {prediction_id}")
            except Exception as db_error:
                logger.error(f"Database error storing prediction: {str(db_error)}")
            
            # Prepare response
            response = PredictionResponse(
                breed=predicted_breed,
                confidence=confidence,
                breed_info=breed_info,
                top_predictions=top_predictions
            )
            
            logger.info(f"Prediction made: {predicted_breed} with {confidence}% confidence")
            
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_base64")
async def predict_breed_base64(request: PredictionRequest):
    """
    Predict breed from base64 encoded image.
    
    Args:
        request: Contains base64 encoded image
        
    Returns:
        PredictionResponse: Contains breed prediction and confidence
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Transform the image
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            top_predictions = []
            
            for prob, idx in zip(top3_probs[0], top3_indices[0]):
                class_idx = idx.item()
                if class_idx < len(CLASS_NAMES):
                    breed_name = CLASS_NAMES[class_idx]
                    confidence = round(prob.item() * 100, 2)
                    
                    top_predictions.append({
                        "breed": breed_name,
                        "confidence": confidence
                    })
            
            # Get the top prediction
            confidence_val, predicted_idx = torch.max(probabilities, 1)
            predicted_class_idx = predicted_idx.item()
            
            if predicted_class_idx >= len(CLASS_NAMES):
                raise HTTPException(status_code=500, detail="Model prediction index out of bounds")
            
            predicted_breed = CLASS_NAMES[predicted_class_idx]
            confidence = round(confidence_val.item() * 100, 2)
            
            # Get breed information
            breed_info = BREED_INFO.get(predicted_breed, {})
            
            # Prepare response
            response = PredictionResponse(
                breed=predicted_breed,
                confidence=confidence,
                breed_info=breed_info,
                top_predictions=top_predictions
            )
            
            logger.info(f"Base64 prediction made: {predicted_breed} with {confidence}% confidence")
            
            return response
            
    except Exception as e:
        logger.error(f"Base64 prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/breeds")
async def get_breeds():
    """Get list of supported breeds."""
    return {"breeds": CLASS_NAMES, "count": len(CLASS_NAMES)}

@app.get("/breed/{breed_name}")
async def get_breed_info(breed_name: str):
    """Get detailed information about a specific breed."""
    # First try to get from hardcoded info
    breed_info = BREED_INFO.get(breed_name)
    
    # Then try to get from database for more details
    db_breed_info = db_manager.get_breed_by_name(breed_name)
    if db_breed_info:
        # Merge database info with hardcoded info
        breed_info = {**db_breed_info, **(breed_info or {}) }
    
    if not breed_info:
        raise HTTPException(status_code=404, detail="Breed not found")
    
    return breed_info

@app.get("/breeds/db")
async def get_all_breeds_from_db():
    """Get all breeds from the database."""
    breeds = db_manager.get_all_breeds()
    return {"breeds": breeds, "count": len(breeds)}

@app.get("/predictions/recent")
async def get_recent_predictions(limit: int = 10):
    """Get recent predictions from the database."""
    predictions = db_manager.get_recent_predictions(limit=limit)
    return {"predictions": predictions, "count": len(predictions)}

@app.get("/predictions/stats")
async def get_prediction_stats():
    """Get prediction statistics."""
    stats = db_manager.get_prediction_statistics()
    return stats

@app.post("/feedback")
async def submit_feedback(
    prediction_id: int,
    is_correct: bool,
    correct_breed: str = None,
    comments: str = None
):
    """Submit feedback for a prediction."""
    try:
        feedback_id = db_manager.insert_user_feedback(
            prediction_id=prediction_id,
            is_correct=is_correct,
            correct_breed=correct_breed,
            comments=comments
        )
        return {"message": "Feedback submitted successfully", "feedback_id": feedback_id}
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)