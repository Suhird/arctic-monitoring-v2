"""
ML API Endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Dict, Any
import torch
import numpy as np
import io
from PIL import Image
from ..ml.model_loader import model_loader

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])

@router.post("/classify")
async def classify_ice_image(file: UploadFile = File(...)):
    """
    Classify uploaded satellite image (Ice Type & Concentration)
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to 256x256 (model input size)
        image = image.resize((256, 256))
        
        # Preprocess
        # Normalize to 0-1 and convert to tensor (NCHW)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0) # [1, 3, 256, 256]
        
        # Load Model
        classifier = model_loader.get_ice_classifier()
        
        # Predict
        with torch.no_grad():
             outputs = classifier(img_tensor)
             probs = torch.softmax(outputs, dim=1)
             confidence, predicted_class = torch.max(probs, 1)
             
             # Classes from training script: 0=Open Water, 1=Thin Ice, 2=Thick Ice
             class_names = ["Open Water", "Thin Ice", "Thick Ice"]
             
             idx = predicted_class.item()
             if 0 <= idx < len(class_names):
                 label = class_names[idx]
             else:
                 label = "Unknown"
                 
             result = {
                 "class": label,
                 "class_id": idx,
                 "confidence": float(confidence.item()),
                 "probabilities": {class_names[i]: float(probs[0][i].item()) for i in range(len(class_names))}
             }

        return {
            "filename": file.filename,
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
