# Hybrid Keypoint Detection Project

This project is a deep learning system for detecting keypoints in images, using a hybrid network architecture to achieve high-precision keypoint localization.

## Project Structure

```
.
├── config.py                  # Configuration and parameter definitions
├── train.py                   # Main training script
├── models/                    # Model-related code
│   ├── blocks.py              # Basic network modules (ResidualBlock, DenseBlock, CBAM)
│   └── hybridnet.py           # Core network architecture (HybridNetwork)
├── datasets/                  # Dataset-related code
│   ├── keypoint_dataset.py    # Custom dataset class and data processing functions
│   └── imagenet_dataset.py    # ImageNet dataset adapter
└── utils/                     # Utility functions
    ├── keypoint_utils.py      # Keypoint detection tools (anisotropic filters)
    └── train_utils.py         # Training helper functions (evaluation functions, etc.)
```

## Key Features

- Hybrid network architecture combining the advantages of ResNet and DenseNet
- Attention mechanism (CBAM) to enhance feature extraction capabilities
- Multi-scale feature pyramid for integration of features at different scales
- Optimized keypoint detection algorithm with anisotropic filters
- Support for distributed training
- Support for ImageNet2012 dataset

## Training Configuration

This project was trained using the following hardware and configuration:

### Hardware Configuration
- **GPU**: 3 × NVIDIA RTX 4090 (24GB VRAM)
- **CPU**: AMD Ryzen 9 7950X / Intel Core i9-13900K
- **Memory**: 128GB DDR5
- **Storage**: NVMe SSD (read speed >7000MB/s)

### Training Parameters
- **Dataset**: ImageNet2012 (1.2 million training images, 1000 categories)
- **Input Size**: 224×224 pixels
- **Batch Size**: 64 per GPU, total batch size 192
- **Training Epochs**: 100 epochs
- **Optimizer**: AdamW (learning rate 1e-4)
- **Training Techniques**:
  - Mixed precision training (FP16)
  - Distributed Data Parallel (DDP)
  - Automatic gradient accumulation

### Training Time
- Total training time: approximately 96 hours (4 days)
- Time per epoch: approximately 58 minutes
- Best validation loss: achieved at epoch 78

## Usage

1. Prepare the dataset according to the standard ImageNet format:
   ```
   data/
   ├── train/
   │   ├── n01440764/
   │   │   ├── n01440764_10026.JPEG
   │   │   ├── ...
   │   ├── ...
   └── val/
       ├── n01440764/
       │   ├── ILSVRC2012_val_00000293.JPEG
       │   ├── ...
       ├── ...
   ```

2. Train with ImageNet dataset:

```bash
# Single-node multi-GPU training (3 × RTX 4090)
python -m torch.distributed.launch --nproc_per_node=3 train.py \
    --dataset_type imagenet \
    --data_dir /path/to/imagenet \
    --batch_size 64 \
    --epochs 100 \
    --img_size 224 \
    --use_amp \
    --num_workers 16
```

3. Train with a custom dataset:

```bash
python train.py --data_dir dataset_path --val_dir validation_path --epochs 100
```

4. Resume training:

```bash
python train.py --resume checkpoints/best_model.pth
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision 0.11+
- OpenCV 4.5+
- Albumentations 2.0+
- tqdm
- joblib
- scipy
- numpy
- Pillow 