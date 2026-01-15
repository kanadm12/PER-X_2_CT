#!/bin/bash

# Installation script for PerX2CT testing dependencies (Python 3.12 compatible)
echo "Installing PerX2CT dependencies for testing (Python 3.12 compatible)..."

# Upgrade pip first
pip install --upgrade pip

# Install core dependencies with Python 3.12 compatible versions
pip install \
    omegaconf==2.0.0 \
    pytorch-lightning \
    albumentations \
    scikit-image \
    scikit-learn \
    scipy \
    numpy \
    Pillow \
    opencv-python \
    imageio \
    h5py \
    tqdm \
    PyYAML \
    matplotlib

# Install medical imaging libraries (Python 3.12 compatible)
pip install \
    nibabel \
    SimpleITK \
    pydicom \
    dicom2nifti

# Install additional dependencies
pip install \
    einops \
    kornia \
    lpips

# Install git-based dependencies if needed
# pip install git+https://github.com/openai/CLIP.git
# pip install git+https://github.com/MIC-DKFZ/batchgenerators.git

echo "======================================"
echo "Installation complete!"
echo "======================================"
