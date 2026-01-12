"""
Ice classification model using ResNet50 (pre-trained + fine-tuned)
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Dict, Tuple
import numpy as np
from PIL import Image


class IceClassifier(nn.Module):
    """
    Ice classification CNN based on ResNet50
    Classifies ice types: open water, thin ice, thick ice
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super(IceClassifier, self).__init__()

        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)

        # Freeze early layers (blocks 1-3)
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False

        # Replace final classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.ice_types = [
            "open_water",
            "thin_ice",
            "thick_ice"
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, 3, 256, 256)

        Returns:
            classification logits
        """
        return self.backbone(x)

    def predict(self, image: np.ndarray) -> Dict[str, any]:
        """
        Predict ice type from image

        Args:
            image: NumPy array (H, W, C) or PIL Image

        Returns:
            Dictionary with ice_type, confidence, and probabilities
        """
        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        input_tensor = transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            classification = self(input_tensor)
            probabilities = torch.softmax(classification, dim=1)
            ice_type_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, ice_type_idx].item()

        return {
            "ice_type": self.ice_types[ice_type_idx],
            "confidence": float(confidence),
            "probabilities": {
                ice_type: float(prob)
                for ice_type, prob in zip(self.ice_types, probabilities[0].numpy())
            }
        }


def create_ice_classifier(weights_path: str = None) -> IceClassifier:
    """
    Factory function to create ice classifier

    Args:
        weights_path: Path to saved weights (optional)

    Returns:
        IceClassifier model
    """
    model = IceClassifier(pretrained=(weights_path is None))

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    model.eval()
    return model
