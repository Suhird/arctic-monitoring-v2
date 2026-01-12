"""
Train LSTM Ice Movement Predictor
Predicts 7-day ice concentration based on 30-day historical sequences
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class IceSequenceDataset(Dataset):
    """Dataset for temporal ice sequences"""

    def __init__(self, data_dir, sequence_length=30, prediction_days=7):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days

        # Load all processed images
        train_images_dir = f"{data_dir}/train/images"
        self.all_samples = sorted([f for f in os.listdir(train_images_dir) if f.endswith('.npy')])

        print(f"Loaded {len(self.all_samples)} samples")
        print(f"Creating sequences: {sequence_length} days → {prediction_days} days")

    def __len__(self):
        # Number of valid sequences we can create
        return max(0, len(self.all_samples) - self.sequence_length - self.prediction_days)

    def __getitem__(self, idx):
        # Load sequence of images (historical)
        sequence_images = []
        for i in range(self.sequence_length):
            sample_idx = idx + i
            sample_path = f"{self.data_dir}/train/images/{self.all_samples[sample_idx]}"
            image = np.load(sample_path)

            # Convert to grayscale concentration map
            concentration = np.mean(image, axis=-1, keepdims=True)  # (256, 256, 1)
            sequence_images.append(concentration)

        # Load future images (targets)
        future_images = []
        for i in range(self.prediction_days):
            sample_idx = idx + self.sequence_length + i
            sample_path = f"{self.data_dir}/train/images/{self.all_samples[sample_idx]}"
            image = np.load(sample_path)
            concentration = np.mean(image, axis=-1, keepdims=True)
            future_images.append(concentration)

        # Convert to tensors
        # Input: (seq_len, channels, height, width)
        sequence = torch.from_numpy(np.array(sequence_images)).permute(0, 3, 1, 2).float()

        # Target: (prediction_days, channels, height, width)
        targets = torch.from_numpy(np.array(future_images)).permute(0, 3, 1, 2).float()

        return sequence, targets


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


class LSTMTrainer:
    """Training manager for LSTM predictor"""

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for sequences, targets in pbar:
            sequences, targets = sequences.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(sequences, future_steps=targets.size(1))

            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for sequences, targets in tqdm(self.val_loader, desc="Validation"):
                sequences, targets = sequences.to(self.device), targets.to(self.device)

                predictions = self.model(sequences, future_steps=targets.size(1))
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self, num_epochs=30, save_dir='../models'):
        """Full training loop"""
        print("=" * 60)
        print("Training LSTM Ice Movement Predictor")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Num epochs: {num_epochs}")
        print()

        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = f"{save_dir}/ice_movement_lstm.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, model_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

        # Plot curves
        self.plot_training_curves(save_dir)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best val loss: {self.best_val_loss:.6f}")
        print(f"Model saved to: {save_dir}/ice_movement_lstm.pth")

    def plot_training_curves(self, save_dir):
        """Plot training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('LSTM Predictor Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/lstm_predictor_training_curves.png", dpi=150)
        print(f"  ✓ Saved training curves")


def main():
    """Main training function"""
    DATA_DIR = "../data/processed"
    BATCH_SIZE = 4  # Small batch size due to memory
    NUM_EPOCHS = 30
    SEQUENCE_LENGTH = 30
    PREDICTION_DAYS = 7
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")
    print()

    # Create datasets
    train_dataset = IceSequenceDataset(
        DATA_DIR,
        sequence_length=SEQUENCE_LENGTH,
        prediction_days=PREDICTION_DAYS
    )

    # Split for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    print()

    # Create model
    model = IceMovementPredictor(input_channels=1, hidden_dim=32, num_layers=2)
    print(f"Model: ConvLSTM Ice Movement Predictor")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Train
    trainer = LSTMTrainer(model, train_loader, val_loader, DEVICE)
    trainer.train(num_epochs=NUM_EPOCHS, save_dir='../models')


if __name__ == "__main__":
    main()
