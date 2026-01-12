# Jupyter Notebooks for Arctic Ice ML

This directory is for Jupyter notebooks for experimentation and visualization.

## Suggested Notebooks

### 1. Data Exploration
`01_data_exploration.ipynb`
- Load and visualize training data
- Analyze ice type distributions
- Plot sample images
- Check data quality

### 2. Model Training
`02_model_training.ipynb`
- Interactive model training
- Real-time training plots
- Hyperparameter tuning
- Model comparison

### 3. Model Evaluation
`03_model_evaluation.ipynb`
- Evaluate on test set
- Confusion matrices
- Per-class accuracy
- Error analysis

### 4. Visualization
`04_predictions_visualization.ipynb`
- Visualize ice predictions
- Compare with ground truth
- Create animated time series
- Generate prediction maps

### 5. Satellite Data Analysis
`05_satellite_data_analysis.ipynb`
- Download Sentinel imagery
- Process SAR data
- Compare satellite sources
- Quality control

## Setup

```bash
# Install Jupyter
pip install jupyter notebook ipywidgets

# Start Jupyter
jupyter notebook
```

## Example Notebook

Create `01_data_exploration.ipynb`:

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

# Load manifest
with open('../data/labeled_ice_imagery/manifest.json', 'r') as f:
    manifest = json.load(f)

print(f"Total samples: {manifest['num_samples']}")

# Load and display sample
sample = manifest['samples'][0]
img = Image.open(f"../data/labeled_ice_imagery/images/{sample['image']}")
label = np.load(f"../data/labeled_ice_imagery/labels/{sample['label']}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(img)
ax1.set_title('Satellite Image')
ax1.axis('off')

ax2.imshow(label, cmap='viridis')
ax2.set_title('Ice Type Label')
ax2.axis('off')

plt.tight_layout()
plt.show()

# Distribution
unique, counts = np.unique(label, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} pixels ({count/label.size*100:.1f}%)")
```

## Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Arctic Ice Datasets](https://nsidc.org/)
- [Sentinel Data Access](https://scihub.copernicus.eu/)
