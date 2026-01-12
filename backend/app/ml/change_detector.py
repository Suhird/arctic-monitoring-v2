"""
Change detection model for temporal ice analysis
Uses Siamese network to detect changes between time periods
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from PIL import Image


class SiameseChangeDetector(nn.Module):
    """
    Siamese network for detecting ice changes between two time periods
    """

    def __init__(self):
        super(SiameseChangeDetector, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Change detection head
        self.change_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()  # Change probability 0-1
        )

        # Upsampling to original resolution
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            img1: Image at time t1 (batch, 3, H, W)
            img2: Image at time t2 (batch, 3, H, W)

        Returns:
            Change map (batch, 1, H, W) with values 0-1
        """
        # Extract features from both images
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)

        # Concatenate features
        combined = torch.cat([features1, features2], dim=1)

        # Detect changes
        change_map = self.change_head(combined)

        # Upsample to original resolution
        change_map = self.upsample(change_map)

        return change_map

    def detect_changes(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Detect changes between two images

        Args:
            img1: Image at time t1
            img2: Image at time t2
            threshold: Change probability threshold

        Returns:
            Binary change map
        """
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Convert to PIL if needed
        if isinstance(img1, np.ndarray):
            img1 = Image.fromarray(img1)
        if isinstance(img2, np.ndarray):
            img2 = Image.fromarray(img2)

        # Transform
        img1_tensor = transform(img1).unsqueeze(0)
        img2_tensor = transform(img2).unsqueeze(0)

        # Inference
        with torch.no_grad():
            change_map = self(img1_tensor, img2_tensor)

        # Convert to binary
        change_map_np = change_map.squeeze().numpy()
        binary_change = (change_map_np > threshold).astype(np.uint8)

        return binary_change


def create_change_detector(weights_path: str = None) -> SiameseChangeDetector:
    """
    Factory function to create change detector

    Args:
        weights_path: Path to saved weights (optional)

    Returns:
        SiameseChangeDetector model
    """
    model = SiameseChangeDetector()

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    model.eval()
    return model
