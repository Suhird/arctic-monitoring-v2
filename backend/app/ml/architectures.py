import torch
import torch.nn as nn
from torchvision import models

# ==========================================
# 1. Ice Classifier (ResNet50)
# ==========================================

class IceClassifier(nn.Module):
    """ResNet50-based ice classifier"""

    def __init__(self, num_classes=3, pretrained=False):
        super(IceClassifier, self).__init__()

        # Load pre-trained ResNet50
        # In inference mode, we usually don't need to download pretrained weights if we are loading our own checkpoint
        # But if the checkpoint was saved as state_dict of the full model, we need the structure.
        self.backbone = models.resnet50(pretrained=pretrained)

        # Replace classifier to match training architecture
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


# ==========================================
# 2. Ice Movement Predictor (ConvLSTM)
# ==========================================

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
            nn.Sigmoid()  # Output 0-1 for concentration
        )

    def forward(self, x, future_steps=7):
        """
        Forward pass

        Args:
            x: Input (batch, seq_len, channels, height, width)
            future_steps: Number of future time steps to predict

        Returns:
            predictions: (batch, future_steps, 1, height, width)
        """
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


# ==========================================
# 3. Siamese Change Detector
# ==========================================

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
