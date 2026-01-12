"""
ML model loading and management
"""
import torch
import os
import logging
from typing import Optional
from ..config import settings
from .architectures import IceClassifier, IceMovementPredictor, SiameseChangeDetector

logger = logging.getLogger(__name__)

class ModelLoader:
    """Singleton class for loading and caching ML models"""

    _instance = None
    _ice_classifier: Optional[IceClassifier] = None
    _ice_predictor: Optional[IceMovementPredictor] = None
    _change_detector: Optional[SiameseChangeDetector] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def get_ice_classifier(self) -> IceClassifier:
        """Load and return ice classification model"""
        if self._ice_classifier is None:
            model_path = os.path.join(settings.MODEL_PATH, settings.ICE_CLASSIFIER_MODEL)
            # Determine if running in backend root or elsewhere
            if not os.path.exists(model_path):
                 # Try absolute path based on CWD if relative fails
                 logging.warning(f"Model path {model_path} not found relative to {os.getcwd()}")
            
            # Initialize architecture
            self._ice_classifier = IceClassifier(num_classes=3, pretrained=False)
            
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    # Load from checkpoint (contains epoch, optimizer, etc.) or state_dict
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self._ice_classifier.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self._ice_classifier.load_state_dict(checkpoint)
                    
                    self._ice_classifier.eval()
                    logger.info(f"✅ Ice classifier loaded from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load weights for Ice Classifier: {e}")
            else:
                logger.warning(f"Ice classifier model file not found at {model_path}. Using random weights.")
                
        return self._ice_classifier

    def get_ice_predictor(self) -> IceMovementPredictor:
        """Load and return ice movement predictor model"""
        if self._ice_predictor is None:
            model_path = os.path.join(settings.MODEL_PATH, settings.ICE_PREDICTOR_MODEL)
            
            # Initialize architecture
            self._ice_predictor = IceMovementPredictor(input_channels=1, hidden_dim=32, num_layers=2)
            
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self._ice_predictor.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self._ice_predictor.load_state_dict(checkpoint)
                    
                    self._ice_predictor.eval()
                    logger.info(f"✅ Ice predictor loaded from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load weights for Ice Predictor: {e}")
            else:
                logger.warning(f"Ice predictor model file not found at {model_path}. Using random weights.")
                
        return self._ice_predictor

    def get_change_detector(self) -> SiameseChangeDetector:
        """Load and return change detector model"""
        if self._change_detector is None:
            model_path = os.path.join(settings.MODEL_PATH, settings.CHANGE_DETECTOR_MODEL)
            
            # Initialize architecture
            self._change_detector = SiameseChangeDetector()
            
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self._change_detector.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self._change_detector.load_state_dict(checkpoint)
                    
                    self._change_detector.eval()
                    logger.info(f"✅ Change detector loaded from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load weights for Change Detector: {e}")
            else:
                logger.warning(f"Change detector model file not found at {model_path}. Using random weights.")
                
        return self._change_detector

    def reload_models(self):
        """Reload all models (useful after training/updates)"""
        self._ice_classifier = None
        self._ice_predictor = None
        self._change_detector = None
        logger.info("All models cleared from cache. Will reload on next request.")


# Global model loader instance
model_loader = ModelLoader()
