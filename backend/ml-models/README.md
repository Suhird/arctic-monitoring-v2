# Arctic Ice ML Training Pipeline

This directory contains the complete machine learning training pipeline for the Arctic Ice Monitoring Platform.

## Quick Start

```bash
# Run the complete training pipeline (one command!)
./train_all.sh
```

This will:
1. Download Arctic ice training data (100 synthetic samples by default)
2. Preprocess data into train/val/test splits
3. Train ResNet50 ice classifier
4. Train LSTM ice movement predictor
5. Train Siamese change detector
6. Save all models to `models/` directory

## Step-by-Step Training

### 1. Download Training Data

```bash
cd data
python download_arctic_data.py
```

**What it does:**
- Generates 100 synthetic Arctic ice imagery samples (256x256 RGB)
- Creates corresponding labels (ice type classifications)
- Generates metadata for each sample
- Creates a manifest.json file

**Output:**
- `labeled_ice_imagery/images/` - Training images
- `labeled_ice_imagery/labels/` - Labels (NumPy arrays)
- `labeled_ice_imagery/metadata/` - Sample metadata
- `labeled_ice_imagery/manifest.json` - Dataset manifest

**For Real Data:**

The script provides links to actual Arctic ice data sources:

**FREE Data Sources:**
1. **NSIDC Sea Ice Concentration**
   - URL: https://nsidc.org/data/g02135/versions/3
   - Data: Daily polar gridded sea ice concentrations
   - Resolution: 25km
   - Format: NetCDF
   - License: Public Domain

2. **Copernicus Sentinel-1/2**
   - URL: https://scihub.copernicus.eu/dhus
   - Data: SAR (Sentinel-1) and optical (Sentinel-2) imagery
   - Resolution: 10-20m
   - Setup: Add `SENTINEL_USERNAME` and `SENTINEL_PASSWORD` to `.env`
   - License: Free (requires registration)

3. **NOAA/NESDIS Ice Charts**
   - URL: https://usicecenter.gov/Products/ArcticData
   - Data: Analyzed ice charts with ice type classifications
   - License: Public Domain

4. **ESA CCI Sea Ice**
   - URL: https://data.ceda.ac.uk/neodc/esacci/sea_ice/data
   - Data: Climate-quality ice concentration
   - License: Free (requires CEDA registration)

5. **MODIS Arctic Imagery**
   - URL: https://worldview.earthdata.nasa.gov/
   - Data: Optical imagery of Arctic regions
   - License: Public Domain

### 2. Preprocess Data

```bash
cd data
python preprocessing.py
```

**What it does:**
- Splits data into train (70%), validation (15%), test (15%)
- Normalizes images to 0-1 range
- Applies data augmentation (flips, rotations, brightness)
- Creates temporal sequences for LSTM training
- Computes class weights for handling imbalanced data
- Saves processed data as NumPy arrays

**Output:**
- `processed/train/` - Training data
- `processed/val/` - Validation data
- `processed/test/` - Test data
- `processed/sequences/` - Temporal sequences
- `processed/dataset_info.json` - Dataset statistics

### 3. Train Ice Classifier (ResNet50)

```bash
cd training
python train_ice_classifier.py
```

**Model Architecture:**
- **Base**: ResNet50 (pre-trained on ImageNet)
- **Fine-tuning**: Last 30 layers trainable
- **Classifier head**:
  - Linear(2048 → 512) + ReLU + Dropout(0.5)
  - Linear(512 → 128) + ReLU
  - Linear(128 → num_classes)

**Classes:**
- 0: Open water
- 1: Thin ice
- 2: Thick ice

**Training Parameters:**
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Cross Entropy (with class weights)
- Scheduler: ReduceLROnPlateau
- Epochs: 20

**Output:**
- `../models/ice_classifier_resnet50.pth` - Trained model
- `../models/ice_classifier_training_curves.png` - Training plots

**Expected Performance:**
- Training accuracy: ~85-90%
- Validation accuracy: ~80-85%
- Inference time: ~50ms per image (CPU)

### 4. Train LSTM Predictor

```bash
python train_lstm_predictor.py
```

**Model Architecture:**
- **Type**: Convolutional LSTM (ConvLSTM)
- **Layers**: 2 stacked ConvLSTM cells
- **Hidden dim**: 32 channels
- **Input**: 30-day historical sequences (256×256)
- **Output**: 7-day future predictions (256×256)

**Training Parameters:**
- Batch size: 4 (memory intensive)
- Learning rate: 0.001
- Optimizer: Adam
- Loss: MSE (Mean Squared Error)
- Gradient clipping: max_norm=1.0
- Epochs: 30

**Output:**
- `../models/ice_movement_lstm.pth` - Trained model
- `../models/lstm_predictor_training_curves.png` - Training plots

**Expected Performance:**
- Validation MSE: <0.01
- 1-day forecast accuracy: ~90%
- 7-day forecast accuracy: ~70%

### 5. Train Change Detector

```bash
python train_change_detector.py
```

**Model Architecture:**
- **Type**: Siamese Convolutional Network
- **Encoder**: Shared feature extractor (4 conv blocks)
- **Decoder**: Change detection head (3 conv layers)
- **Input**: Two images (t1 and t2)
- **Output**: Binary change map (256×256)

**Training Parameters:**
- Batch size: 8
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Binary Cross Entropy
- Time gap: 7 days
- Epochs: 20

**Output:**
- `../models/change_detector.pth` - Trained model
- `../models/change_detector_training_curves.png` - Training plots

**Expected Performance:**
- Validation accuracy: ~85-90%
- Precision: ~80%
- Recall: ~85%

## Using Trained Models

### Copy Models to Backend

```bash
# From ml-models directory
cp models/*.pth ../backend/app/models/
```

### Restart Backend to Load Models

```bash
docker-compose restart backend
```

### Test Models via API

```python
import requests

# Classify ice type
response = requests.post('http://localhost:8000/api/v1/classify',
    files={'image': open('ice_sample.png', 'rb')})
print(response.json())
# Output: {"ice_type": "thick_ice", "confidence": 0.92}

# Get 7-day prediction
response = requests.get('http://localhost:8000/api/v1/predictions/7day',
    params={'min_lon': -180, 'min_lat': 60, 'max_lon': -120, 'max_lat': 85})
print(response.json())
```

## Dataset Structure

```
data/
├── labeled_ice_imagery/          # Raw data
│   ├── images/                   # Training images (PNG)
│   ├── labels/                   # Labels (NumPy)
│   ├── metadata/                 # Sample metadata (JSON)
│   └── manifest.json             # Dataset manifest
├── processed/                    # Preprocessed data
│   ├── train/                    # Training split
│   │   ├── images/               # Normalized images
│   │   ├── labels/               # Processed labels
│   │   └── info.json             # Split statistics
│   ├── val/                      # Validation split
│   ├── test/                     # Test split
│   ├── sequences/                # Temporal sequences for LSTM
│   └── dataset_info.json         # Overall dataset info
└── download_arctic_data.py       # Data download script
```

## Model Files

```
models/
├── ice_classifier_resnet50.pth    # Ice type classifier (ResNet50)
├── ice_movement_lstm.pth          # Ice movement predictor (LSTM)
├── change_detector.pth            # Change detector (Siamese)
├── *_training_curves.png          # Training visualizations
└── README.md                      # Model documentation
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- torch==2.1.0
- torchvision==0.16.0
- numpy==1.24.3
- matplotlib==3.8.2
- tqdm==4.66.1
- Pillow==10.1.0
- rasterio==1.3.9 (for satellite imagery)
- sentinelsat==1.2.1 (for Sentinel data)

## GPU Training

Models train much faster on GPU:

**CPU Training Time:**
- Ice Classifier: ~2 hours (20 epochs)
- LSTM Predictor: ~4 hours (30 epochs)
- Change Detector: ~1.5 hours (20 epochs)

**GPU Training Time (NVIDIA RTX 3090):**
- Ice Classifier: ~15 minutes
- LSTM Predictor: ~30 minutes
- Change Detector: ~10 minutes

**Enable GPU:**
```python
# Training scripts automatically detect CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Customization

### Increase Training Data

```python
# In download_arctic_data.py
downloader.download_all(num_synthetic_samples=1000)  # Default: 100
```

### Adjust Model Architecture

```python
# In train_ice_classifier.py
model = IceClassifier(
    num_classes=5,  # Add more ice types
    pretrained=True
)

# In train_lstm_predictor.py
model = IceMovementPredictor(
    hidden_dim=64,  # Increase capacity (default: 32)
    num_layers=3    # Add more layers (default: 2)
)
```

### Modify Training Parameters

```python
# Batch size
BATCH_SIZE = 64  # Default: 32 (classifier), 4 (LSTM)

# Learning rate
lr = 0.0001  # Default: 0.001

# Epochs
NUM_EPOCHS = 50  # Default: 20-30
```

## Evaluation

### Evaluate Trained Models

```python
# Test ice classifier
python -c "
from training.train_ice_classifier import *
model = IceClassifier(num_classes=3)
model.load_state_dict(torch.load('models/ice_classifier_resnet50.pth')['model_state_dict'])
# Run evaluation on test set
"
```

### Visualize Predictions

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load test image
img = Image.open('data/processed/test/images/sample_0000.npy')

# Run inference
# ... (see backend/app/ml/ice_classifier.py for inference code)

# Plot results
plt.imshow(img)
plt.title(f'Predicted: {ice_type} ({confidence*100:.1f}%)')
plt.show()
```

## Troubleshooting

### Out of Memory Errors

```python
# Reduce batch size
BATCH_SIZE = 2  # For LSTM on CPU

# Use gradient accumulation
accumulation_steps = 4
```

### Low Accuracy

1. **Increase training data**: Generate more synthetic samples or add real data
2. **Data augmentation**: Already enabled in preprocessing
3. **Learning rate**: Try 0.0001 or 0.01
4. **Class imbalance**: Class weights are automatically computed

### Slow Training

1. **Use GPU**: 10-30x faster than CPU
2. **Reduce image size**: Resize to 128×128 (faster but less accurate)
3. **Multi-GPU**: Use `torch.nn.DataParallel`

## Next Steps

1. **Add Real Satellite Data**:
   - Register at data sources listed above
   - Download actual Arctic imagery
   - Label using ice charts

2. **Improve Models**:
   - Use EfficientNet instead of ResNet50
   - Add attention mechanisms to LSTM
   - Implement U-Net for segmentation

3. **Deploy Models**:
   - Models are already integrated in backend
   - API endpoints ready to use
   - Add model versioning

4. **Monitor Performance**:
   - Track prediction accuracy over time
   - Compare with actual ice conditions
   - Retrain periodically with new data

## Citations

If using this ML pipeline, please cite:

```bibtex
@software{arctic_ice_ml,
  title = {Arctic Ice Monitoring Platform - ML Training Pipeline},
  year = {2024},
  author = {Your Organization},
  url = {https://github.com/your-org/arctic-ice-platform}
}
```

## License

MIT License - See LICENSE file

## Support

For issues:
- GitHub Issues: <repo-url>/issues
- Documentation: /docs
- Email: support@your-org.com
