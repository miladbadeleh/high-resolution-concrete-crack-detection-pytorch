# High-Resolution Concrete Crack Detection using ResNet Architecture

## Overview
This project implements a deep learning model based on the ResNet-18 architecture to detect cracks in high-resolution concrete images. The model is built using PyTorch and is designed for binary classification (crack vs. no-crack) with high accuracy.

## Key Components

### 1. Data Preparation
- **Dataset Structure**: Images should be placed in a single directory, with filenames containing 'crack' for positive examples
- **Image Preprocessing**: Images are resized to 224×224 pixels and normalized using ImageNet statistics
- **Custom Dataset Class**: The `ConcreteCrackDataset` class handles image loading and label assignment

### 2. Model Architecture (ResNet-18)
- **Pre-trained Backbone**: Uses ResNet-18 with pre-trained weights from ImageNet
- **Modified Final Layer**: The original fully connected layer is replaced with a single-unit layer for binary classification
- **Activation**: Sigmoid activation is applied to the output for probability estimation

### 3. Training Configuration
- **Loss Function**: Binary Cross Entropy Loss (BCELoss)
- **Optimizer**: Adam optimizer with learning rate 0.001
- **Batch Size**: 32 images per batch
- **Epochs**: 10 (configurable)
- **Device**: Automatically uses GPU if available (CUDA), otherwise falls back to CPU

### 4. Model Saving
- The trained model weights are saved in PyTorch format (high-resolution-crack_detection_model-resnet-arch.pth) for future use

## Requirements
- Python 3.x
- PyTorch
- torchvision
- Pillow (PIL)
- NumPy

## Usage
1. Organize your images in a directory with filenames containing 'crack' for positive examples
2. Update the path in the notebook: `img_dir='path_to_images'`
3. Run the notebook cells sequentially to:
   - Load and preprocess the data
   - Initialize the ResNet-18 model
   - Train the model
   - Save the trained model weights

## Notes
- The model expects input images of size 224×224 pixels with 3 color channels (RGB)
- Label assignment is based on filename containing 'crack' (case-sensitive)
- The number of epochs and batch size can be adjusted based on your dataset size and computational resources

## Implementation Details
- The implementation uses PyTorch's DataLoader for efficient batch processing and includes:

- Image normalization using ImageNet mean and std

- Automatic GPU utilization when available

- Simple filename-based labeling system

- ResNet-18 transfer learning with fine-tuning

## Installation
```bash
pip install torch torchvision pillow numpy
