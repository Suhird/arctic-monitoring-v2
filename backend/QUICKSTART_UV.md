# 🚀 Quick Start with UV

Get up and running in **2 minutes**!

## One-Line Setup

```bash
./setup_uv.sh
```

That's it! The script will guide you through the setup.

## What Gets Installed?

When you run the setup script, you'll be asked to choose:

1. **Full installation** ✅ (Recommended)
   - Everything you need: Backend + ML + Jupyter
   - Size: ~3-4 GB
   - Time: ~60 seconds

2. **Backend only**
   - FastAPI, Database, API development
   - Size: ~2 GB
   - Time: ~40 seconds

3. **ML only**
   - Training models, no backend
   - Size: ~2.5 GB
   - Time: ~50 seconds

4. **ML + Dev**
   - Training + Jupyter notebooks (no backend)
   - Size: ~2.7 GB
   - Time: ~50 seconds

## After Setup

### Activate Your Environment

Every time you start a new terminal session:

```bash
source .venv/bin/activate
```

### Train ML Models

```bash
cd ml-models

# Download sample data (generates 100 synthetic images)
cd data
python download_arctic_data.py

# Train all models
cd ..
python train_all.py

# Or train individually
python training/train_ice_classifier.py
python training/train_lstm_predictor.py
python training/train_change_detector.py
```

### Use Jupyter Notebooks

```bash
cd ml-models/notebooks
jupyter notebook
```

Then explore:
- `01_data_exploration.ipynb` - Visualize your data
- `02_model_training.ipynb` - Train models interactively
- `03_model_evaluation.ipynb` - Evaluate model performance
- `04_predictions_visualization.ipynb` - Beautiful prediction visualizations
- `05_satellite_data_analysis.ipynb` - Real satellite data integration

### Run Backend (Local Development)

```bash
cd backend

# Start PostgreSQL and Redis with Docker
docker-compose up -d postgres redis

# Run FastAPI
uvicorn app.main:app --reload
```

Visit: http://localhost:8000/docs

### Run Full Stack (Production)

```bash
docker-compose up -d
```

Starts:
- PostgreSQL + PostGIS: `localhost:5432`
- Redis: `localhost:6379`
- Backend API: `localhost:8000`
- Frontend: `localhost:3000`

## Common Commands

```bash
# Activate environment
source .venv/bin/activate

# Install new package
uv pip install <package-name>

# Update all packages
uv pip install --upgrade -e ".[all]"

# List installed packages
uv pip list

# Deactivate environment
deactivate
```

## Troubleshooting

### UV command not found

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### PyTorch installation slow

```bash
# Use CPU version (faster)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Import errors

```bash
# Make sure venv is activated
source .venv/bin/activate

# Reinstall
uv pip install --reinstall -e ".[all]"
```

## Project Structure

```
arctic-ice-monitoring-v2/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── api/         # API endpoints
│   │   ├── ml/          # ML models
│   │   ├── models/      # Database models
│   │   └── services/    # Business logic
│   └── requirements.txt
├── ml-models/           # ML training
│   ├── data/            # Dataset scripts
│   ├── training/        # Training scripts
│   ├── notebooks/       # Jupyter notebooks
│   └── models/          # Saved models
├── frontend/            # React frontend
├── pyproject.toml       # UV dependencies ⭐
├── setup_uv.sh          # Quick setup script ⭐
└── UV_SETUP.md          # Detailed UV guide ⭐
```

## Next Steps

1. ✅ Run `./setup_uv.sh`
2. ✅ Activate venv: `source .venv/bin/activate`
3. ✅ Download data: `cd ml-models/data && python download_arctic_data.py`
4. ✅ Train models: `cd .. && python train_all.py`
5. ✅ Explore notebooks: `cd notebooks && jupyter notebook`
6. ✅ Run backend: `cd ../../backend && uvicorn app.main:app --reload`

## Need Help?

- **UV Setup Details**: [UV_SETUP.md](UV_SETUP.md)
- **ML Training Guide**: [ml-models/ML_TRAINING_GUIDE.md](ml-models/ML_TRAINING_GUIDE.md)
- **Full Documentation**: [README.md](README.md)
- **API Documentation**: http://localhost:8000/docs (after running backend)

## Why UV?

- **10-100x faster** than pip ⚡
- **Reliable** dependency resolution
- **Disk efficient** with global cache
- **Compatible** with existing Python tools
- **Modern** Python packaging standards

**Installation time comparison:**
- pip: ~5-10 minutes ⏳
- poetry: ~3-6 minutes ⏱️
- **uv: ~30-60 seconds** 🚀
