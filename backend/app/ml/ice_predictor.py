"""
Ice movement prediction model using LSTM
Predicts 7-day ice concentration forecast based on historical data
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for spatial-temporal prediction
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int]):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, input_tensor: torch.Tensor, cur_state: Tuple[torch.Tensor, torch.Tensor]):
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


class IceMovementPredictor(nn.Module):
    """
    LSTM-based ice movement predictor
    Input: 30 days of historical ice concentration data
    Output: 7 days of future predictions
    """

    def __init__(self, input_channels: int = 1, hidden_dim: int = 64, num_layers: int = 2):
        super(IceMovementPredictor, self).__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ConvLSTM layers
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(
                input_dim=input_channels if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=(3, 3)
            )
            for i in range(num_layers)
        ])

        # Output projection (must match trained model: 16 channels, not 32)
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Output 0-1 for concentration
        )

    def forward(self, x: torch.Tensor, future_steps: int = 7) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, seq_len, channels, height, width)
            future_steps: Number of days to predict

        Returns:
            Predictions (batch, future_steps, 1, height, width)
        """
        batch_size, seq_len, _, height, width = x.size()

        # Initialize hidden states
        h_states = []
        c_states = []
        for _ in range(self.num_layers):
            h_states.append(torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device))
            c_states.append(torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device))

        # Process historical sequence
        for t in range(seq_len):
            input_t = x[:, t, :, :, :]

            for layer_idx in range(self.num_layers):
                if layer_idx == 0:
                    h_states[layer_idx], c_states[layer_idx] = self.convlstm_layers[layer_idx](
                        input_t, (h_states[layer_idx], c_states[layer_idx])
                    )
                else:
                    h_states[layer_idx], c_states[layer_idx] = self.convlstm_layers[layer_idx](
                        h_states[layer_idx - 1], (h_states[layer_idx], c_states[layer_idx])
                    )

        # Generate future predictions
        predictions = []
        last_output = x[:, -1, :, :, :]

        for _ in range(future_steps):
            # Use last prediction/observation as input
            for layer_idx in range(self.num_layers):
                if layer_idx == 0:
                    h_states[layer_idx], c_states[layer_idx] = self.convlstm_layers[layer_idx](
                        last_output, (h_states[layer_idx], c_states[layer_idx])
                    )
                else:
                    h_states[layer_idx], c_states[layer_idx] = self.convlstm_layers[layer_idx](
                        h_states[layer_idx - 1], (h_states[layer_idx], c_states[layer_idx])
                    )

            # Generate prediction
            pred = self.output_conv(h_states[-1])
            predictions.append(pred)
            last_output = pred

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # (batch, future_steps, 1, H, W)

        return predictions

    def predict(self, historical_data: np.ndarray, future_days: int = 7) -> np.ndarray:
        """
        Predict future ice concentration

        Args:
            historical_data: Historical concentration data (days, height, width) or (days, channels, height, width)
            future_days: Number of days to predict

        Returns:
            Predictions array (future_days, height, width)
        """
        # Prepare input
        if historical_data.ndim == 3:
            historical_data = np.expand_dims(historical_data, axis=1)  # Add channel dim

        input_tensor = torch.from_numpy(historical_data).float().unsqueeze(0)  # Add batch dim

        # Inference
        with torch.no_grad():
            predictions = self(input_tensor, future_steps=future_days)

        # Convert to numpy and scale to 0-100
        predictions_np = predictions.squeeze(0).squeeze(1).numpy() * 100

        return predictions_np


def create_ice_predictor(weights_path: str = None) -> IceMovementPredictor:
    """
    Factory function to create ice movement predictor

    Args:
        weights_path: Path to saved weights (optional)

    Returns:
        IceMovementPredictor model
    """
    model = IceMovementPredictor()

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    model.eval()
    return model
