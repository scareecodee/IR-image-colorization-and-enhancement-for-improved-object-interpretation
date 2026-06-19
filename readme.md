1. Clone/Create Project Structure

# Create project directory
mkdir thermal_colorization
cd thermal_colorization

# Create all necessary directories
mkdir -p config data/{raw,processed} models losses training utils evaluation inference checkpoints logs outputs

# Verify directory structure
ls -la
# Should show: config data models losses training utils evaluation inference checkpoints logs outputs

2. Create Virtual Environment
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation
which python
# Should point to: .../thermal_colorization/venv/bin/python


3. Install Dependencies
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA is not detected, install PyTorch with CUDA support:
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


4. Create Configuration File
# Create the config file
cat > config/config.yaml << 'EOF'
# Configuration for Thermal IR Colorization Pipeline
# Using Bands: B2 (Blue), B3 (Green), B4 (Red), B10 (Thermal)

# Data Configuration
data:
  raw_dir: './data/raw'
  processed_dir: './data/processed'
  patch_size: 256
  stride: 128
  img_size: [256, 256]
  batch_size: 8
  num_workers: 4
  rgb_bands: [4, 3, 2]
  thermal_band: 10

# Model Configuration
model:
  generator:
    in_channels: 1
    out_channels: 3
    base_channels: 64
    num_res_blocks: 9
    use_semantic: false
  discriminator:
    input_channels: 4
    base_channels: 64
    n_layers: 3

# Training Configuration
training:
  g_lr: 0.0002
  d_lr: 0.0002
  num_epochs: 50
  lr_step: 10
  lr_gamma: 0.5
  lambda_adv: 0.03
  lambda_perceptual: 1.0
  lambda_tv: 1.0
  checkpoint_interval: 5

# Evaluation Configuration
evaluation:
  batch_size: 1
  num_samples: 100

# Inference Configuration
inference:
  tile_size: 256
  overlap: 32

# Logging
logging:
  log_dir: './logs'
  checkpoint_dir: './checkpoints'
  output_dir: './outputs'
  sample_interval: 5
EOF

echo "✅ Config file created: config/config.yaml"

📁 Data Setup Commands

5. Prepare Your Data Directory Structure

# Create data directories
mkdir -p data/raw data/processed

# Your data should be organized like this:
# data/raw/
# ├── LC08_L1TP_143042_20200101/
# │   ├── LC08_L1TP_143042_20200101_B2.TIF
# │   ├── LC08_L1TP_143042_20200101_B3.TIF
# │   ├── LC08_L1TP_143042_20200101_B4.TIF
# │   ├── LC08_L1TP_143042_20200101_B10.TIF
# │   └── LC08_L1TP_143042_20200101_metadata.txt
# └── ...

🚀 Pipeline Execution Commands

7. Run Complete Pipeline

# Run everything (preprocess + train + evaluate + inference)
python main.py --phase all

# Or run specific phases
python main.py --phase preprocess   # Data preprocessing only
python main.py --phase train        # Model training only
python main.py --phase evaluate     # Model evaluation only
python main.py --phase inference    # Run inference only

8. Monitor Training with TensorBoard
# Start TensorBoard in a new terminal
tensorboard --logdir logs --port 6006

# Open in browser: http://localhost:6006

# Or run in background
nohup tensorboard --logdir logs --port 6006 > tensorboard.log 2>&1 &

9. Quick Test Commands
# Test if everything is working
python -c "
import torch
import numpy as np
import rasterio
import cv2
import albumentations as A
print('✅ All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Test dataset loading
python -c "
from data.dataset import create_dataloaders
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
dataloaders = create_dataloaders(
    config['data']['processed_dir'],
    batch_size=2,
    img_size=(256, 256)
)
print(f'✅ Train samples: {len(dataloaders[\"train\"].dataset)}')
print(f'✅ Val samples: {len(dataloaders[\"val\"].dataset)}')
"

# Test model creation
python -c "
from models.generator import CoarseToFineGenerator
model = CoarseToFineGenerator()
print(f'✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters')
"


Check Training Progress
# View GPU memory usage
watch -n 1 nvidia-smi

# Check disk usage
du -sh data/processed/

# Check latest logs
tail -f logs/training.log

# Monitor system resources
htop



# Export to ONNX
python -c "
import torch
from models.generator import CoarseToFineGenerator

model = CoarseToFineGenerator()
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['generator_state_dict'])
model.eval()

dummy_input = torch.randn(1, 1, 256, 256)
torch.onnx.export(
    model, dummy_input,
    'outputs/model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
print('✅ Model exported to outputs/model.onnx')
"

Export Model for Deployment

# Convert to TensorRT (requires NVIDIA GPU)
python -c "
import torch
from models.generator import CoarseToFineGenerator
import torch_tensorrt

model = CoarseToFineGenerator()
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['generator_state_dict'])
model.eval()

trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=(1, 1, 256, 256), dtype=torch.float32)],
    enabled_precisions={torch.float16}
)
torch.jit.save(trt_model, 'outputs/model_trt.jit')
"



📝 Full Command Summary
# One-command setup (if starting fresh)
curl -s https://raw.githubusercontent.com/your-repo/thermal_colorization/setup.sh | bash

# Quick reference commands
alias pc='python main.py --config config/config.yaml'
alias tb='tensorboard --logdir logs --port 6006'
alias gpu='nvidia-smi'
alias clean='find . -type f -name "*.pyc" -delete && find . -type d -name "__pycache__" -delete'

# Run specific steps
pc --phase preprocess
pc --phase train
pc --phase evaluate
pc --phase inference