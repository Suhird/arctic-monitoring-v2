"""
Train Siamese Change Detector
Detects changes in ice concentration between two time periods
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class IceChangeDataset(Dataset):
    """Dataset for ice change detection"""

    def __init__(self, data_dir, split='train', time_gap=7):
        self.data_dir = f"{data_dir}/{split}"
        self.time_gap = time_gap

        # Load all samples
        self.samples = sorted([f for f in os.listdir(f"{self.data_dir}/images") if f.endswith('.npy')])

        print(f"Loaded {len(self.samples)} {split} samples (time gap: {time_gap} days)")

    def __len__(self):
        return max(0, len(self.samples) - self.time_gap)

    def __getitem__(self, idx):
        # Image at time t1
        img1_path = f"{self.data_dir}/images/{self.samples[idx]}"
        img1 = np.load(img1_path)

        # Image at time t2 (time_gap days later)
        img2_path = f"{self.data_dir}/images/{self.samples[idx + self.time_gap]}"
        img2 = np.load(img2_path)

        # Load labels
        label1_path = f"{self.data_dir}/labels/{self.samples[idx]}"
        label1 = np.load(label1_path)

        label2_path = f"{self.data_dir}/labels/{self.samples[idx + self.time_gap]}"
        label2 = np.load(label2_path)

        # Create change mask (binary: changed or not)
        change_mask = (label1 != label2).astype(np.float32)

        # Convert to tensors
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()  # HWC -> CHW
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        change_mask = torch.from_numpy(change_mask).unsqueeze(0).float()  # Add channel dim

        return img1, img2, change_mask


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


class ChangeDetectorTrainer:
    """Training manager for change detector"""

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function (Binary Cross Entropy)
        self.criterion = nn.BCELoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for img1, img2, change_mask in pbar:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            change_mask = change_mask.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(img1, img2)

            loss = self.criterion(predictions, change_mask)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0

        with torch.no_grad():
            for img1, img2, change_mask in tqdm(self.val_loader, desc="Validation"):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                change_mask = change_mask.to(self.device)

                predictions = self.model(img1, img2)
                loss = self.criterion(predictions, change_mask)

                total_loss += loss.item()

                # Calculate accuracy
                predicted_binary = (predictions > 0.5).float()
                accuracy = (predicted_binary == change_mask).float().mean()
                total_accuracy += accuracy.item()

        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_accuracy / len(self.val_loader)

        return avg_loss, avg_accuracy * 100

    def train(self, num_epochs=20, save_dir='../models'):
        """Full training loop"""
        print("=" * 60)
        print("Training Siamese Change Detector")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Num epochs: {num_epochs}")
        print()

        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            self.scheduler.step(val_loss)

            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val Acc:    {val_acc:.2f}%")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = f"{save_dir}/change_detector.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, model_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

        # Plot curves
        self.plot_training_curves(save_dir)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best val loss: {self.best_val_loss:.6f}")
        print(f"Best val acc: {max(self.val_accuracies):.2f}%")
        print(f"Model saved to: {save_dir}/change_detector.pth")

    def plot_training_curves(self, save_dir):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('BCE Loss')
        ax1.set_title('Change Detector Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Change Detection Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/change_detector_training_curves.png", dpi=150)
        print(f"  ✓ Saved training curves")


def main():
    """Main training function"""
    DATA_DIR = "../data/processed"
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    TIME_GAP = 7  # Compare images 7 days apart
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")
    print()

    # Create datasets
    train_dataset = IceChangeDataset(DATA_DIR, split='train', time_gap=TIME_GAP)
    val_dataset = IceChangeDataset(DATA_DIR, split='val', time_gap=TIME_GAP)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Create model
    model = SiameseChangeDetector()
    print(f"Model: Siamese Change Detector")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Train
    trainer = ChangeDetectorTrainer(model, train_loader, val_loader, DEVICE)
    trainer.train(num_epochs=NUM_EPOCHS, save_dir='../models')


if __name__ == "__main__":
    main()
