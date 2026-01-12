"""
Data preprocessing for Arctic ice imagery
Prepares data for training ice classification and prediction models
"""

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split


class IceDataPreprocessor:
    """Preprocess Arctic ice imagery for ML training"""

    def __init__(self, data_dir="./labeled_ice_imagery", output_dir="./processed"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/train/images", exist_ok=True)
        os.makedirs(f"{output_dir}/train/labels", exist_ok=True)
        os.makedirs(f"{output_dir}/val/images", exist_ok=True)
        os.makedirs(f"{output_dir}/val/labels", exist_ok=True)
        os.makedirs(f"{output_dir}/test/images", exist_ok=True)
        os.makedirs(f"{output_dir}/test/labels", exist_ok=True)

    def load_manifest(self):
        """Load the data manifest"""
        manifest_path = f"{self.data_dir}/manifest.json"
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        return manifest

    def normalize_image(self, image):
        """Normalize image to 0-1 range"""
        image = np.array(image).astype(np.float32) / 255.0
        return image

    def augment_image(self, image, label):
        """Data augmentation for training"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            label = np.fliplr(label)

        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flipud(image)
            label = np.flipud(label)

        # Random rotation (90, 180, 270 degrees)
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)

        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 1)

        return image, label

    def compute_class_weights(self, labels):
        """Compute class weights for handling imbalanced data"""
        unique, counts = np.unique(labels, return_counts=True)
        total = np.sum(counts)

        weights = {}
        for cls, count in zip(unique, counts):
            weights[int(cls)] = total / (len(unique) * count)

        print(f"Class distribution:")
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples ({count/total*100:.1f}%)")

        print(f"\nClass weights:")
        for cls, weight in weights.items():
            print(f"  Class {cls}: {weight:.3f}")

        return weights

    def split_dataset(self, samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train/val/test"""
        assert train_ratio + val_ratio + test_ratio == 1.0

        # First split: train and temp
        train_samples, temp_samples = train_test_split(
            samples, test_size=(val_ratio + test_ratio), random_state=42
        )

        # Second split: val and test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=(1 - val_ratio_adjusted), random_state=42
        )

        return train_samples, val_samples, test_samples

    def process_sample(self, sample, augment=False):
        """Process a single sample"""
        # Load image
        image_path = f"{self.data_dir}/images/{sample['image']}"
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Load label
        label_path = f"{self.data_dir}/labels/{sample['label']}"
        label = np.load(label_path)

        # Normalize image
        image = self.normalize_image(image)

        # Augment if training
        if augment:
            image, label = self.augment_image(image, label)

        return image, label

    def create_temporal_sequences(self, samples, sequence_length=30):
        """
        Create temporal sequences for LSTM training
        Groups samples by time to create sequences
        """
        print(f"Creating temporal sequences (length={sequence_length})...")

        # For synthetic data, we'll create random sequences
        # In production, group by actual temporal order

        sequences = []
        for i in range(0, len(samples) - sequence_length + 1, sequence_length // 2):
            sequence = samples[i:i + sequence_length]
            if len(sequence) == sequence_length:
                sequences.append(sequence)

        print(f"Created {len(sequences)} sequences")
        return sequences

    def save_processed_data(self, samples, split_name, augment=False):
        """Save processed data for a split"""
        print(f"Processing {split_name} set ({len(samples)} samples)...")

        all_labels = []

        for i, sample in enumerate(tqdm(samples)):
            # Process sample
            image, label = self.process_sample(sample, augment=augment)

            # Save processed image
            image_output = f"{self.output_dir}/{split_name}/images/sample_{i:04d}.npy"
            np.save(image_output, image)

            # Save processed label
            label_output = f"{self.output_dir}/{split_name}/labels/sample_{i:04d}.npy"
            np.save(label_output, label)

            all_labels.append(label)

        # Compute statistics
        all_labels = np.concatenate([l.flatten() for l in all_labels])
        class_weights = self.compute_class_weights(all_labels)

        # Save split info
        split_info = {
            "split": split_name,
            "num_samples": len(samples),
            "class_weights": class_weights,
            "augmented": augment
        }

        with open(f"{self.output_dir}/{split_name}/info.json", 'w') as f:
            json.dump(split_info, f, indent=2)

        return split_info

    def preprocess_all(self):
        """Preprocess entire dataset"""
        print("=" * 60)
        print("Arctic Ice Data Preprocessing")
        print("=" * 60)
        print()

        # Load manifest
        manifest = self.load_manifest()
        samples = manifest['samples']

        print(f"Total samples: {len(samples)}")
        print()

        # Split dataset
        print("Splitting dataset...")
        train_samples, val_samples, test_samples = self.split_dataset(samples)

        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val:   {len(val_samples)} samples")
        print(f"  Test:  {len(test_samples)} samples")
        print()

        # Process each split
        train_info = self.save_processed_data(train_samples, "train", augment=True)
        print()

        val_info = self.save_processed_data(val_samples, "val", augment=False)
        print()

        test_info = self.save_processed_data(test_samples, "test", augment=False)
        print()

        # Create temporal sequences for LSTM training
        print("Creating temporal sequences for prediction model...")
        train_sequences = self.create_temporal_sequences(train_samples, sequence_length=30)

        sequences_dir = f"{self.output_dir}/sequences"
        os.makedirs(sequences_dir, exist_ok=True)

        # Save sequence indices
        with open(f"{sequences_dir}/train_sequences.json", 'w') as f:
            json.dump([{"samples": [s['image'] for s in seq]} for seq in train_sequences], f)

        print(f"Saved {len(train_sequences)} temporal sequences")
        print()

        # Save overall dataset info
        dataset_info = {
            "train": train_info,
            "val": val_info,
            "test": test_info,
            "num_sequences": len(train_sequences),
            "sequence_length": 30
        }

        with open(f"{self.output_dir}/dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)

        print("=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print()
        print("Next steps:")
        print("  1. Train ice classifier: python training/train_ice_classifier.py")
        print("  2. Train LSTM predictor: python training/train_lstm_predictor.py")
        print()

        return dataset_info


def main():
    """Main preprocessing function"""
    preprocessor = IceDataPreprocessor(
        data_dir="./labeled_ice_imagery",
        output_dir="./processed"
    )

    dataset_info = preprocessor.preprocess_all()

    print(f"Dataset ready for training!")
    print(f"Train samples: {dataset_info['train']['num_samples']}")
    print(f"Val samples: {dataset_info['val']['num_samples']}")
    print(f"Test samples: {dataset_info['test']['num_samples']}")


if __name__ == "__main__":
    main()
