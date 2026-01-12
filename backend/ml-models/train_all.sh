#!/bin/bash

# Master script to download data, preprocess, and train all models

echo "=========================================="
echo "Arctic Ice ML Training Pipeline"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
echo ""

# Step 1: Download data
echo "=========================================="
echo "Step 1: Downloading Arctic Ice Data"
echo "=========================================="
cd data
python download_arctic_data.py
echo ""

# Step 2: Preprocess data
echo "=========================================="
echo "Step 2: Preprocessing Data"
echo "=========================================="
python preprocessing.py
cd ..
echo ""

# Step 3: Train Ice Classifier
echo "=========================================="
echo "Step 3: Training Ice Classifier (ResNet50)"
echo "=========================================="
cd training
python train_ice_classifier.py
echo ""

# Step 4: Train LSTM Predictor
echo "=========================================="
echo "Step 4: Training LSTM Predictor"
echo "=========================================="
python train_lstm_predictor.py
echo ""

# Step 5: Train Change Detector
echo "=========================================="
echo "Step 5: Training Change Detector"
echo "=========================================="
python train_change_detector.py
echo ""

# Summary
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Trained models saved in: models/"
echo "  - ice_classifier_resnet50.pth"
echo "  - ice_movement_lstm.pth"
echo "  - change_detector.pth"
echo ""
echo "Copy these models to: ../backend/app/models/"
echo "  cp models/*.pth ../backend/app/models/"
echo ""
echo "Then restart the backend:"
echo "  docker-compose restart backend"
echo ""
