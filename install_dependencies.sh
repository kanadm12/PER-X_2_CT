#!/bin/bash

# Installation script for PerX2CT testing dependencies
echo "Installing PerX2CT dependencies for testing..."

# Upgrade pip first
pip install --upgrade pip

# Install PyTorch and related packages (adjust CUDA version as needed)
# Using PyTorch with CUDA 11.3 as example - modify if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install \
    omegaconf==2.0.0 \
    pytorch-lightning==1.0.8 \
    albumentations==1.1.0 \
    scikit-image==0.19.1 \
    scikit-learn==1.0.2 \
    scipy==1.7.3 \
    numpy==1.21.5 \
    Pillow==9.0.0 \
    opencv-python==4.5.5.62 \
    imageio==2.13.5 \
    h5py==3.6.0 \
    tqdm==4.62.3 \
    PyYAML==6.0 \
    matplotlib==3.5.1

# Install medical imaging libraries
pip install \
    nibabel==3.2.2 \
    SimpleITK==2.1.1 \
    pydicom==2.2.2 \
    dicom2nifti==2.3.2 \
    MedPy==0.4.0

# Install additional dependencies
pip install \
    einops \
    kornia \
    lpips

# Install git-based dependencies
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/MIC-DKFZ/batchgenerators.git

echo "======================================"
echo "Installation complete!"
echo "======================================"
