# UV Setup Guide

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

## Installation

### Install UV

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

## Project Setup

### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd /Users/suhird/Desktop/ideas/arctic-ice-monitoring-v2

# Create venv with uv
uv venv

# Activate the virtual environment
# macOS/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies

**Option A: Install all dependencies (recommended for full development)**
```bash
uv pip install -e ".[all]"
```

**Option B: Install only backend dependencies**
```bash
uv pip install -e ".[backend]"
```

**Option C: Install only ML training dependencies**
```bash
uv pip install -e ".[ml]"
```

**Option D: Install for development (includes Jupyter)**
```bash
uv pip install -e ".[ml,dev]"
```

**Option E: Custom combination**
```bash
# Backend + Development tools
uv pip install -e ".[backend,dev]"

# ML + Development tools
uv pip install -e ".[ml,dev]"
```

### 3. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.11+

# Check installed packages
uv pip list

# Test FastAPI import (if backend installed)
python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"

# Test PyTorch import (if ML installed)
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Common Commands

### Add a new dependency
```bash
# Add to main dependencies
uv pip install <package-name>

# Add to dev dependencies
uv pip install --dev <package-name>
```

### Update dependencies
```bash
uv pip install --upgrade -e ".[all]"
```

### Sync dependencies (clean install)
```bash
uv pip install --reinstall -e ".[all]"
```

### Generate requirements.txt (if needed)
```bash
uv pip freeze > requirements-frozen.txt
```

## Quick Start Workflows

### For Backend Development
```bash
# 1. Create venv
uv venv

# 2. Activate
source .venv/bin/activate

# 3. Install backend dependencies
uv pip install -e ".[backend,dev]"

# 4. Run backend
cd backend
uvicorn app.main:app --reload
```

### For ML Training
```bash
# 1. Create venv
uv venv

# 2. Activate
source .venv/bin/activate

# 3. Install ML dependencies
uv pip install -e ".[ml,dev]"

# 4. Download data
cd ml-models/data
python download_arctic_data.py

# 5. Train models
cd ../
python train_all.py

# 6. Or use Jupyter
jupyter notebook notebooks/
```

### For Full Stack Development
```bash
# 1. Create venv
uv venv

# 2. Activate
source .venv/bin/activate

# 3. Install everything
uv pip install -e ".[all]"

# 4. Start services
docker-compose up -d postgres redis

# 5. Run backend
cd backend
uvicorn app.main:app --reload

# 6. Run frontend (in another terminal)
cd frontend
npm install
npm start
```

## Why UV?

- **10-100x faster** than pip
- **Disk space efficient** with global cache
- **Reliable** dependency resolution
- **Compatible** with pip and existing tools
- **Modern** Python packaging standards

## Troubleshooting

### UV not found after installation
```bash
# Add to PATH (macOS/Linux)
export PATH="$HOME/.cargo/bin:$PATH"

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
```

### PyTorch installation issues
```bash
# For CPU-only (faster installation)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### TensorFlow installation issues
```bash
# Use pip for TensorFlow (better compatibility)
pip install tensorflow==2.14.0
```

### PostGIS/psycopg2 binary issues
```bash
# If psycopg2-binary fails, use system package:
# macOS:
brew install postgresql
uv pip install psycopg2

# Ubuntu/Debian:
sudo apt-get install libpq-dev
uv pip install psycopg2
```

## Performance Comparison

```bash
# Benchmark: Install all dependencies

# pip:          ~5-10 minutes
# pip-tools:    ~4-8 minutes
# poetry:       ~3-6 minutes
# uv:           ~30-60 seconds ⚡
```

## Additional Resources

- UV Documentation: https://github.com/astral-sh/uv
- Python Packaging Guide: https://packaging.python.org/
- Project README: [README.md](README.md)
- ML Training Guide: [ml-models/ML_TRAINING_GUIDE.md](ml-models/ML_TRAINING_GUIDE.md)
