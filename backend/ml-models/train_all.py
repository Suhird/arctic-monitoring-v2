"""
Master training script - Runs complete ML pipeline
Works on all platforms (Windows, Mac, Linux)
"""

import os
import sys
import subprocess


def run_command(command, description):
    """Run a command and display output"""
    print("\n" + "=" * 60)
    print(description)
    print("=" * 60)
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed")
        sys.exit(1)
    print()


def main():
    """Run complete training pipeline"""
    print("\n" + "=" * 60)
    print("Arctic Ice ML Training Pipeline")
    print("=" * 60)
    print()

    # Step 1: Download data
    run_command(
        "cd data && python download_arctic_data.py",
        "Step 1: Downloading Arctic Ice Data"
    )

    # Step 2: Preprocess data
    run_command(
        "cd data && python preprocessing.py",
        "Step 2: Preprocessing Data"
    )

    # Step 3: Train ice classifier
    run_command(
        "cd training && python train_ice_classifier.py",
        "Step 3: Training Ice Classifier (ResNet50)"
    )

    # Step 4: Train LSTM predictor
    run_command(
        "cd training && python train_lstm_predictor.py",
        "Step 4: Training LSTM Predictor"
    )

    # Step 5: Train change detector
    run_command(
        "cd training && python train_change_detector.py",
        "Step 5: Training Change Detector"
    )

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print()
    print("Trained models saved in: models/")
    print("  - ice_classifier_resnet50.pth")
    print("  - ice_movement_lstm.pth")
    print("  - change_detector.pth")
    print()
    print("Next steps:")
    print("  1. Copy models to backend:")
    print("     cp models/*.pth ../backend/app/models/")
    print()
    print("  2. Restart backend:")
    print("     docker-compose restart backend")
    print()


if __name__ == "__main__":
    main()
