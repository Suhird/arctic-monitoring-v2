"""
Train ResNet50 Ice Classifier
Fine-tunes ResNet50 on Arctic ice imagery for ice type classification
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt


class IceDataset(Dataset):
    """PyTorch dataset for ice imagery"""

    def __init__(self, data_dir, split='train'):
        self.data_dir = f"{data_dir}/{split}"
        self.image_dir = f"{self.data_dir}/images"
        self.label_dir = f"{self.data_dir}/labels"

        # Load split info
        with open(f"{self.data_dir}/info.json", 'r') as f:
            self.info = json.load(f)

        # Get all samples
        self.samples = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])

        print(f"Loaded {len(self.samples)} {split} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]

        # Load image (already normalized 0-1)
        image_path = f"{self.image_dir}/{sample_name}"
        image = np.load(image_path)
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW

        # Load label
        label_path = f"{self.label_dir}/{sample_name}"
        label = np.load(label_path)

        # Convert label to class label (use mode of the patch)
        unique, counts = np.unique(label, return_counts=True)
        dominant_class = unique[np.argmax(counts)]

        return image, torch.tensor(dominant_class, dtype=torch.long)


class IceClassifier(nn.Module):
    """ResNet50-based ice classifier"""

    def __init__(self, num_classes=3, pretrained=True):
        super(IceClassifier, self).__init__()

        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)

        # Freeze early layers
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False

        # Replace classifier
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


class Trainer:
    """Training manager for ice classifier"""

    def __init__(self, model, train_loader, val_loader, device, num_classes=3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes

        # Loss function with class weights
        # In production, use actual class weights from dataset
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # Confusion matrix
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update confusion matrix
                for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                    confusion[t, p] += 1

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy, confusion

    def train(self, num_epochs=20, save_dir='../models'):
        """Full training loop"""
        print("=" * 60)
        print("Training Ice Classifier")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Num epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print()

        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc, confusion = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            # Print confusion matrix
            print(f"\nConfusion Matrix:")
            print(confusion)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = f"{save_dir}/ice_classifier_resnet50.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, model_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

        # Plot training curves
        self.plot_training_curves(save_dir)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best val acc: {max(self.val_accuracies):.2f}%")
        print(f"Model saved to: {save_dir}/ice_classifier_resnet50.pth")

    def plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/ice_classifier_training_curves.png", dpi=150)
        print(f"  ✓ Saved training curves")


def main():
    """Main training function"""
    # Configuration
    DATA_DIR = "../data/processed"
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    NUM_CLASSES = 3  # open_water, thin_ice, thick_ice
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")
    print()

    # Create datasets
    train_dataset = IceDataset(DATA_DIR, split='train')
    val_dataset = IceDataset(DATA_DIR, split='val')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = IceClassifier(num_classes=NUM_CLASSES, pretrained=True)
    print(f"Model: ResNet50-based Ice Classifier")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        num_classes=NUM_CLASSES
    )

    # Train
    trainer.train(num_epochs=NUM_EPOCHS, save_dir='../models')


if __name__ == "__main__":
    main()
