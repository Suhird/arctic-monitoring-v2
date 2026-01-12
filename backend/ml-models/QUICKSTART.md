# ML Training Quick Start

Get your Arctic ice ML models trained in 3 commands!

## Option 1: One-Line Training (Recommended)

```bash
# Install dependencies and run complete pipeline
pip install -r requirements.txt && python train_all.py
```

This will:
1. Download 100 synthetic Arctic ice samples
2. Preprocess data (train/val/test split)
3. Train ResNet50 ice classifier (~15-30 min)
4. Train LSTM predictor (~30-60 min)
5. Train change detector (~10-20 min)
6. Save all models to `models/` directory

**Total time**: ~1-2 hours (CPU) or ~15-30 minutes (GPU)

## Option 2: Step-by-Step

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (100 synthetic samples)
cd data
python download_arctic_data.py
cd ..

# 3. Preprocess data
cd data
python preprocessing.py
cd ..

# 4. Train models
cd training
python train_ice_classifier.py    # ~15-30 min
python train_lstm_predictor.py     # ~30-60 min
python train_change_detector.py    # ~10-20 min
cd ..
```

## Option 3: Bash Script (Linux/Mac)

```bash
chmod +x train_all.sh
./train_all.sh
```

## Using Trained Models

### Copy to Backend

```bash
# From ml-models directory
cp models/*.pth ../backend/app/models/

# Restart backend to load models
cd ..
docker-compose restart backend
```

### Test via API

```bash
# Get 7-day prediction
curl "http://localhost:8000/api/v1/predictions/7day?min_lon=-180&min_lat=60&max_lon=-120&max_lat=85"
```

## Expected Output

After training completes, you'll have:

```
ml-models/
├── models/
│   ├── ice_classifier_resnet50.pth          # ~95 MB
│   ├── ice_movement_lstm.pth                # ~50 MB
│   ├── change_detector.pth                  # ~180 MB
│   ├── ice_classifier_training_curves.png   # Training plots
│   ├── lstm_predictor_training_curves.png
│   └── change_detector_training_curves.png
└── data/
    ├── labeled_ice_imagery/                 # Raw data (100 samples)
    └── processed/                           # Preprocessed data
```

## Training Data Sources

### Synthetic Data (Default)
- **Size**: 100 samples (256×256 RGB images)
- **Training time**: 1-2 hours
- **Purpose**: Quick start and testing
- **Accuracy**: ~75-80% (baseline)

### Real Satellite Data (Production)

Download real Arctic ice data from these **FREE** sources:

1. **NSIDC Sea Ice Concentration** (Recommended!)
   ```bash
   # Public domain, no registration
   wget https://noaadata.apps.nsidc.org/NOAA/G02202_V4/north/daily/2023/seaice_conc_daily_nh_20230101_f17_v04r00.nc
   ```
   - URL: https://nsidc.org/data/g02135
   - Resolution: 25km
   - Coverage: Daily, Arctic-wide

2. **Sentinel-1/2** (ESA Copernicus)
   ```bash
   # Add to .env file:
   SENTINEL_USERNAME=your_username
   SENTINEL_PASSWORD=your_password
   ```
   - Register: https://scihub.copernicus.eu/dhus
   - Resolution: 10-20m
   - Coverage: 5-6 day revisit

3. **NOAA Ice Charts**
   - URL: https://usicecenter.gov/Products/ArcticData
   - Format: Shapefiles with ice classifications
   - No registration required

With real data:
- **Training samples**: 1000-10,000+
- **Training time**: 2-8 hours
- **Accuracy**: 85-95%

## Customization

### More Synthetic Samples

```python
# Edit data/download_arctic_data.py
# Change line:
downloader.download_all(num_synthetic_samples=500)  # Default: 100
```

### Fewer Epochs (Faster Training)

```python
# Edit training scripts
NUM_EPOCHS = 10  # Default: 20-30
```

### Smaller Models (Lower Memory)

```python
# In train_ice_classifier.py
BATCH_SIZE = 16  # Default: 32

# In train_lstm_predictor.py
BATCH_SIZE = 2   # Default: 4
model = IceMovementPredictor(hidden_dim=16, num_layers=1)  # Default: 32, 2
```

## GPU Training

**Automatic GPU detection** - scripts use GPU if available!

Check GPU availability:
```python
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

**Speed comparison** (100 samples, 20 epochs):
- CPU (Intel i7): ~2 hours
- GPU (RTX 3060): ~15 minutes
- GPU (RTX 3090): ~8 minutes

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size in training scripts
# ice_classifier: BATCH_SIZE = 16 (default: 32)
# lstm_predictor: BATCH_SIZE = 2 (default: 4)
```

### Missing Dependencies

```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### "No samples found"

```bash
# Re-download data
cd data
rm -rf labeled_ice_imagery processed
python download_arctic_data.py
python preprocessing.py
```

### Low Accuracy

1. Increase training samples:
   ```python
   num_synthetic_samples=500  # in download_arctic_data.py
   ```

2. Train longer:
   ```python
   NUM_EPOCHS = 50  # in training scripts
   ```

3. Use real satellite data (see sources above)

## Next Steps

1. **Increase data size**: Download more samples
2. **Add real data**: Use Sentinel or NSIDC data
3. **Fine-tune hyperparameters**: Adjust learning rate, batch size
4. **Deploy models**: Copy to backend and test via API
5. **Monitor performance**: Track accuracy on new data

## Support

- **Documentation**: See [README.md](README.md) for detailed guide
- **Issues**: GitHub Issues
- **Models not working?**: Ensure backend is restarted after copying models

---

**Ready to train?**

```bash
pip install -r requirements.txt && python train_all.py
```

Good luck! 🚀
