# IR-image-colorization-and-enhancement-for-improved-object-interpretation
AI-powered infrared satellite image enhancement and colorization using super-resolution and GANs to generate realistic RGB imagery for improved object interpretation.


# Thermal IR Colorization for ISRO Hackathon 2026

## Problem Statement
Colorize Landsat 8/9 thermal infrared imagery (Band 10) using RGB bands (2, 3, 4) to create realistic RGB images.

## Key Features
- ✅ Uses only Bands 2, 3, 4, 10 (no Band 11 dependency)
- ✅ Coarse-to-Fine Generator with Attention
- ✅ Perceptual loss for better visual quality
- ✅ Full preprocessing pipeline
- ✅ Tiled inference for large satellite scenes
- ✅ Comprehensive evaluation metrics

## Quick Start

### 1. Setup
```bash
# Clone repository
git clone <your-repo>
cd thermal_colorization_isro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


thermal_colorization_isro/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── preprocessing.py
│   └── data_loader.py
│
├── models/
│   ├── __init__.py
│   ├── generator.py
│   ├── discriminator.py
│   ├── super_resolution.py
│   └── attention.py
│
├── losses/
│   ├── __init__.py
│   └── losses.py
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── utils.py
│
├── inference/
│   ├── __init__.py
│   └── predict.py
│
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── visualization.py
│   └── geospatial.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── requirements.txt
├── setup.py
├── README.md
├── .gitignore
└── main.py
