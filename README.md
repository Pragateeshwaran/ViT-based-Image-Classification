# Classification Vision Transformer

## Overview
This project implements a Vision Transformer (ViT) model for classifying Cats and Dogs images. The implementation includes custom self-attention mechanisms, data preprocessing, and training pipeline.

## Dependencies
- Python 3.8+
- PyTorch
- torchvision
- OpenCV (cv2)
- NumPy
- Matplotlib
- tqdm

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd squamoscan-vit
```

2. Install required packages:
```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
```

## Directory Structure
```
squamoscan-vit/
├── data/
│   ├── train/
│   └── test/
├── model.py           # ViT model implementation
├── train.py          # Training and evaluation script
├── DataProcessing.py  # Data preprocessing and loading
└── README.md
```

## Usage
1. Prepare your dataset:
- Place training images in `data/train/` with subfolders for each class
- Place testing images in `data/test/` with subfolders for each class

2. Preprocess images:
```bash
python DataProcessing.py
```

3. Train the model and visualize results:
```bash
python train.py
```

## Model Architecture
- Patch Size: 16x16
- Embedding Dimension: 512
- Transformer Blocks: 8
- Attention Heads: 8
- Input Size: 256x256x3
- Classes: 2 (binary classification)

## Training Configuration
- Optimizer: AdamW (lr=0.0005, weight_decay=0.01)
- Loss: CrossEntropy with label smoothing (0.1)
- Scheduler: CosineAnnealingLR
- Epochs: 100
- Batch Size: 32

## Data Preprocessing
- Training: Random flips, rotations, color jitter
- Testing: Resize and normalization
- All images resized to 256x256 with padding

## Results
The script outputs:
- Training loss per epoch
- Test accuracy
- Visualization of 6 sample predictions
