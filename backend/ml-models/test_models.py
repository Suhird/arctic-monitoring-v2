"""
Test trained models with sample predictions and visualizations
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# Import model architectures
import sys
sys.path.append(str(Path(__file__).parent / "training"))


class IceClassifier(nn.Module):
    """ResNet50-based ice classifier"""
    def __init__(self, num_classes=3, pretrained=False):
        super(IceClassifier, self).__init__()

        # Load ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)

        # Replace classifier (must match training architecture)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell"""
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )


class IceMovementPredictor(nn.Module):
    """LSTM-based ice movement predictor"""
    def __init__(self, input_channels=1, hidden_dim=32, num_layers=2):
        super(IceMovementPredictor, self).__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ConvLSTM layers
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(
                input_dim=input_channels if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=3
            )
            for i in range(num_layers)
        ])

        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, future_steps=7):
        batch_size, seq_len, _, height, width = x.size()

        # Initialize hidden states
        hidden_states = []
        for layer in self.convlstm_layers:
            hidden_states.append(layer.init_hidden(batch_size, (height, width)))

        # Process historical sequence
        for t in range(seq_len):
            input_t = x[:, t, :, :, :]

            for layer_idx in range(self.num_layers):
                if layer_idx == 0:
                    h, c = self.convlstm_layers[layer_idx](input_t, hidden_states[layer_idx])
                else:
                    h, c = self.convlstm_layers[layer_idx](hidden_states[layer_idx - 1][0], hidden_states[layer_idx])
                hidden_states[layer_idx] = (h, c)

        # Generate future predictions
        predictions = []
        last_output = x[:, -1, :, :, :]

        for _ in range(future_steps):
            for layer_idx in range(self.num_layers):
                if layer_idx == 0:
                    h, c = self.convlstm_layers[layer_idx](last_output, hidden_states[layer_idx])
                else:
                    h, c = self.convlstm_layers[layer_idx](hidden_states[layer_idx - 1][0], hidden_states[layer_idx])
                hidden_states[layer_idx] = (h, c)

            # Generate prediction
            pred = self.output_conv(hidden_states[-1][0])
            predictions.append(pred)
            last_output = pred

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # (batch, future_steps, 1, H, W)

        return predictions


class SiameseChangeDetector(nn.Module):
    """Siamese network for ice change detection"""
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
            nn.Sigmoid()
        )

        # Upsample to original resolution
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, img1, img2):
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


class ModelTester:
    """Test all trained models"""

    def __init__(self, data_dir="./data/labeled_ice_imagery", models_dir="./models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")
        print()

        # Load manifest
        with open(self.data_dir / "manifest.json", 'r') as f:
            self.manifest = json.load(f)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.class_names = {
            0: "Open Water",
            1: "Thin Ice",
            2: "Thick Ice"
        }

        self.colors = {
            0: [0, 0, 255],      # Blue - Open Water
            1: [135, 206, 250],  # Light Blue - Thin Ice
            2: [255, 255, 255]   # White - Thick Ice
        }

    def load_models(self):
        """Load all trained models"""
        print("=" * 60)
        print("Loading Trained Models")
        print("=" * 60)

        # 1. Ice Classifier
        print("Loading Ice Classifier...")
        self.classifier = IceClassifier(num_classes=3).to(self.device)
        classifier_path = self.models_dir / "ice_classifier_resnet50.pth"
        if classifier_path.exists():
            checkpoint = torch.load(classifier_path, map_location=self.device)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            self.classifier.eval()
            print(f"✓ Loaded: {classifier_path}")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        else:
            print(f"✗ Not found: {classifier_path}")
            self.classifier = None

        # 2. LSTM Predictor
        print("Loading LSTM Predictor...")
        self.lstm = IceMovementPredictor(input_channels=1, hidden_dim=32, num_layers=2).to(self.device)
        lstm_path = self.models_dir / "ice_movement_lstm.pth"
        if lstm_path.exists():
            checkpoint = torch.load(lstm_path, map_location=self.device)
            self.lstm.load_state_dict(checkpoint['model_state_dict'])
            self.lstm.eval()
            print(f"✓ Loaded: {lstm_path}")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        else:
            print(f"✗ Not found: {lstm_path}")
            self.lstm = None

        # 3. Change Detector
        print("Loading Change Detector...")
        self.change_detector = SiameseChangeDetector().to(self.device)
        detector_path = self.models_dir / "change_detector.pth"
        if detector_path.exists():
            checkpoint = torch.load(detector_path, map_location=self.device)
            self.change_detector.load_state_dict(checkpoint['model_state_dict'])
            self.change_detector.eval()
            print(f"✓ Loaded: {detector_path}")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        else:
            print(f"✗ Not found: {detector_path}")
            self.change_detector = None

        print()

    def test_ice_classifier(self, num_samples=5):
        """Test ice classifier on sample images"""
        if self.classifier is None:
            print("Ice classifier not loaded, skipping...")
            return

        print("=" * 60)
        print("Testing Ice Classifier")
        print("=" * 60)

        # Select random samples
        samples = np.random.choice(self.manifest['samples'], num_samples, replace=False)

        fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)

        with torch.no_grad():
            for i, sample in enumerate(samples):
                # Load image
                img_path = self.data_dir / "images" / sample['image']
                img = Image.open(img_path).convert('RGB')

                # Load ground truth label
                label_path = self.data_dir / "labels" / sample['label']
                gt_label = np.load(label_path)

                # Prepare input
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)

                # Predict
                outputs = self.classifier(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()

                # Visualize
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"Input Image\n{sample['image'][:20]}...", fontsize=8)
                axes[0, i].axis('off')

                # Create colored prediction visualization
                pred_viz = np.zeros((256, 256, 3), dtype=np.uint8)
                pred_viz[:, :] = self.colors[pred_class]

                axes[1, i].imshow(pred_viz)
                axes[1, i].set_title(
                    f"Prediction: {self.class_names[pred_class]}\n"
                    f"Confidence: {confidence:.2%}",
                    fontsize=8
                )
                axes[1, i].axis('off')

                print(f"Sample {i+1}: {self.class_names[pred_class]} ({confidence:.2%})")

        plt.tight_layout()
        plt.savefig('test_results_classifier.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: test_results_classifier.png")
        print()

    def test_lstm_predictor(self, num_samples=3, sequence_length=30, future_steps=7):
        """Test LSTM ice movement predictor"""
        if self.lstm is None:
            print("LSTM predictor not loaded, skipping...")
            return

        print("=" * 60)
        print("Testing LSTM Ice Movement Predictor")
        print("=" * 60)

        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        with torch.no_grad():
            for i in range(num_samples):
                # Create synthetic sequence (grayscale concentration maps)
                # In production, this would be actual historical data
                sequence = []
                for t in range(sequence_length):
                    # Load a random sample and convert to grayscale
                    sample_idx = np.random.randint(0, len(self.manifest['samples']))
                    img_path = self.data_dir / "images" / self.manifest['samples'][sample_idx]['image']
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img_np = np.array(img) / 255.0  # Normalize to 0-1
                    sequence.append(img_np)

                # Convert to tensor: (1, seq_len, 1, H, W)
                sequence_tensor = torch.from_numpy(np.array(sequence)).unsqueeze(0).unsqueeze(2).float().to(self.device)

                # Predict future
                predictions = self.lstm(sequence_tensor, future_steps=future_steps)  # (1, future_steps, 1, H, W)

                # Visualize
                # Last historical frame
                axes[i, 0].imshow(sequence[-1], cmap='gray', vmin=0, vmax=1)
                axes[i, 0].set_title(f"Last Historical Frame (t={sequence_length})", fontsize=9)
                axes[i, 0].axis('off')

                # First predicted frame
                pred_1 = predictions[0, 0, 0].cpu().numpy()
                axes[i, 1].imshow(pred_1, cmap='gray', vmin=0, vmax=1)
                axes[i, 1].set_title(f"Prediction t+1", fontsize=9)
                axes[i, 1].axis('off')

                # Middle predicted frame
                mid_idx = future_steps // 2
                pred_mid = predictions[0, mid_idx, 0].cpu().numpy()
                axes[i, 2].imshow(pred_mid, cmap='gray', vmin=0, vmax=1)
                axes[i, 2].set_title(f"Prediction t+{mid_idx+1}", fontsize=9)
                axes[i, 2].axis('off')

                # Last predicted frame
                pred_last = predictions[0, -1, 0].cpu().numpy()
                axes[i, 3].imshow(pred_last, cmap='gray', vmin=0, vmax=1)
                axes[i, 3].set_title(f"Prediction t+{future_steps}", fontsize=9)
                axes[i, 3].axis('off')

                print(f"Sequence {i+1}: Predicted {future_steps} future timesteps from {sequence_length}-day history")

        plt.tight_layout()
        plt.savefig('test_results_lstm.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: test_results_lstm.png")
        print()

    def test_change_detector(self, num_samples=3):
        """Test change detection on image pairs"""
        if self.change_detector is None:
            print("Change detector not loaded, skipping...")
            return

        print("=" * 60)
        print("Testing Siamese Change Detector")
        print("=" * 60)

        # Select random pairs
        samples = self.manifest['samples']

        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        with torch.no_grad():
            for i in range(num_samples):
                # Select two random images
                idx1, idx2 = np.random.choice(len(samples), 2, replace=False)

                # Load images
                img1_path = self.data_dir / "images" / samples[idx1]['image']
                img2_path = self.data_dir / "images" / samples[idx2]['image']

                img1 = Image.open(img1_path).convert('RGB')
                img2 = Image.open(img2_path).convert('RGB')

                img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
                img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)

                # Detect changes
                change_map = self.change_detector(img1_tensor, img2_tensor)
                change_map = change_map.squeeze().cpu().numpy()

                # Visualize
                axes[i, 0].imshow(img1)
                axes[i, 0].set_title(f"Image 1 (t)", fontsize=10)
                axes[i, 0].axis('off')

                axes[i, 1].imshow(img2)
                axes[i, 1].set_title(f"Image 2 (t+Δt)", fontsize=10)
                axes[i, 1].axis('off')

                # Change heatmap
                im = axes[i, 2].imshow(change_map, cmap='hot', vmin=0, vmax=1)
                axes[i, 2].set_title(f"Change Heatmap\nMax: {change_map.max():.3f}", fontsize=10)
                axes[i, 2].axis('off')
                plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

                # Binary change mask (threshold at 0.5)
                binary_change = (change_map > 0.5).astype(np.uint8)
                axes[i, 3].imshow(binary_change, cmap='gray')
                axes[i, 3].set_title(
                    f"Change Detection\n{(binary_change.sum() / binary_change.size * 100):.1f}% changed",
                    fontsize=10
                )
                axes[i, 3].axis('off')

                print(f"Pair {i+1}: Detected {binary_change.sum() / binary_change.size * 100:.1f}% change")

        plt.tight_layout()
        plt.savefig('test_results_change_detector.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: test_results_change_detector.png")
        print()

    def run_all_tests(self):
        """Run all model tests"""
        self.load_models()

        print("=" * 60)
        print("Running Model Tests")
        print("=" * 60)
        print()

        self.test_ice_classifier(num_samples=5)
        self.test_lstm_predictor(num_samples=3)
        self.test_change_detector(num_samples=3)

        print("=" * 60)
        print("All Tests Complete!")
        print("=" * 60)
        print()
        print("Generated visualizations:")
        print("  - test_results_classifier.png")
        print("  - test_results_lstm.png")
        print("  - test_results_change_detector.png")
        print()


def main():
    """Main testing function"""
    # Change to ml-models directory
    os.chdir(Path(__file__).parent)

    tester = ModelTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
