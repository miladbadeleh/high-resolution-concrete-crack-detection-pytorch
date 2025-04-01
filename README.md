# High-Resolution Concrete Crack Detection using ResNet Architecture

## Overview
This project implements a deep learning model based on the ResNet-18 architecture to detect cracks in high-resolution concrete images. The model is built using TensorFlow and Keras, and is designed for binary classification (crack vs. no-crack) with high accuracy.

## Key Components

### 1. Data Preparation
- **Dataset Structure**: Images should be organized in separate train and validation directories, each containing subdirectories for 'crack' and 'no_crack' classes
- **Image Preprocessing**: Images are resized to 224×224 pixels and normalized to values between 0.0 and 1.0
- **Data Augmentation**: Training images are augmented with random rotations, shifts, shears, zooms, and flips to improve model generalization

### 2. Model Architecture (ResNet-18)
- **Initial Convolution**: 7×7 convolution with stride 2 followed by max pooling
- **Residual Blocks**: Four groups of residual blocks with [64, 128, 256, 512] filters respectively
- **Shortcut Connections**: Bypass connections that help mitigate vanishing gradients in deep networks
- **Final Layers**: Global average pooling followed by a dense layer with sigmoid activation for binary classification

### 3. Training Configuration
- **Loss Function**: Binary cross-entropy (suitable for binary classification)
- **Optimizer**: Adam optimizer
- **Metrics**: Accuracy
- **Batch Size**: 32 images per batch
- **Epochs**: 10 (configurable)

### 4. Evaluation
- Model performance is evaluated on a separate validation set during training
- Training and validation metrics are tracked for each epoch

### 5. Model Saving
- The trained model is saved in HDF5 format (resnet18_crack_detection.h5) for future use

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pillow (PIL)
- OpenCV (optional, for additional image processing)

## Usage
- Organize your dataset into train and val directories with class subfolders

- Update the paths in the notebook: train_dir = 'path_to_train_images', val_dir = 'path_to_val_images'

- Run the notebook cells sequentially to:

- - Build the ResNet-18 model

- - Preprocess and augment the data

- - Train the model

- - Save the trained model

## Notes
- The model expects input images of size 224×224 pixels with 3 color channels (RGB)

- For best results, ensure your training dataset is balanced between crack and no-crack examples

- The number of epochs and batch size can be adjusted based on your dataset size and computational resources

## Installation
```bash
pip install tensorflow numpy pillow

