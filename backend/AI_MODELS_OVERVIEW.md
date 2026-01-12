# AI Models in Arctic Ice Monitoring Platform

## Overview

Your platform uses **3 Deep Learning models** for Arctic ice analysis and prediction. All models have been trained and saved in `/ml-models/models/`.

---

## 🤖 Model 1: Ice Classifier (ResNet50)

### Purpose
Classifies ice types and estimates ice concentration from satellite imagery.

### Architecture
- **Base**: ResNet50 (pretrained on ImageNet, fine-tuned for Arctic ice)
- **Input**: Satellite images (256×256 RGB)
- **Output**:
  - Ice type classification (5 classes)
  - Ice concentration percentage (0-100%)

### Ice Types
1. **Open Water** - No ice coverage
2. **Thin Ice** - First-year ice, fragile
3. **Thick Ice** - Multi-year ice, stable
4. **Multi-Year Ice** - Oldest, hardest ice
5. **Pack Ice** - Dense ice formations

### Model File
`models/ice_classifier_resnet50.pth` (213 MB)

### Current Usage
**NOT CURRENTLY INTEGRATED** - Frontend shows demo data from database.

---

## 🔮 Model 2: Ice Movement Predictor (ConvLSTM)

### Purpose
Predicts 7-day future ice concentration based on 30 days of historical data.

### Architecture
- **Type**: Convolutional LSTM (spatial-temporal)
- **Input**: 30 days of historical ice concentration grids
- **Output**: 7 days of future ice concentration predictions
- **Hidden Layers**: 2-layer ConvLSTM with 32/64 hidden dimensions

### How It Works
1. Takes 30 days of past ice concentration maps
2. Learns spatial patterns (ice shapes) and temporal patterns (ice movement)
3. Generates 7 daily predictions with confidence scores

### Model File
`models/ice_movement_lstm.pth` (1.4 MB)

### Current Usage
**NOT CURRENTLY INTEGRATED** - Frontend shows static demo predictions from database.

---

## 🔍 Model 3: Change Detector (Siamese Network)

### Purpose
Detects changes in ice coverage between two time periods (e.g., week-over-week).

### Architecture
- **Type**: Siamese Convolutional Neural Network
- **Input**: Two satellite images (before/after)
- **Output**: Change probability map (0-1)

### Use Cases
- Alert generation for rapid ice melts
- Track seasonal ice retreat/advance
- Identify anomalous changes

### Model File
`models/change_detector.pth` (86 MB)

### Current Usage
**NOT INTEGRATED** - No API endpoint exists yet.

---

## 📊 Training Results

All models have been trained and tested. See visualization images:
- `ml-models/test_results_classifier.png`
- `ml-models/test_results_lstm.png`
- `ml-models/test_results_change_detector.png`

---

## ⚠️ Current Status

### What's Working
✅ All 3 models are trained and saved
✅ Model architectures defined in `/backend/app/ml/`
✅ Model loader infrastructure exists

### What's NOT Working
❌ **Models are not used for predictions** - API returns static demo data
❌ **No real-time inference** - Need to integrate models into API endpoints
❌ **No model server** - Models loaded from `/ml-models/models/` not `/app/models`

---

## 🔧 Integration Plan

To use your trained models for real predictions:

### 1. Copy Models to Backend
```bash
cp ml-models/models/*.pth backend/models/
```

### 2. Update Predictions API
Modify `/backend/app/api/predictions.py` to:
- Load the LSTM predictor model
- Fetch 30 days of historical ice data
- Run model inference to generate predictions
- Return model predictions instead of database demo data

### 3. Add Ice Classification Endpoint
Create new endpoint `/api/v1/ice/classify` to:
- Accept satellite image upload
- Use ResNet50 classifier
- Return ice type and concentration

### 4. Add Change Detection Endpoint
Create `/api/v1/ice/changes` to:
- Accept two timestamps
- Fetch images for those dates
- Run change detector
- Return change map

---

## 📈 Next Steps

1. **Test Models** - Run `ml-models/test_models.py` to verify models work
2. **Integrate LSTM Predictor** - Make 7-day predictions use real model
3. **Add Classification Endpoint** - Allow users to upload images for analysis
4. **Deploy Model Server** - Consider using TorchServe for production

---

## 🧪 Testing Your Models

See the test script we'll create to verify model predictions work correctly.
