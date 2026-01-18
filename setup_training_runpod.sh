#!/bin/bash
# ===============================================
# PerX2CT Training Setup Script for RunPod
# ===============================================
# This script sets up the environment and prepares
# your custom dataset for training on RunPod.
#
# Usage:
#   chmod +x setup_training_runpod.sh
#   ./setup_training_runpod.sh
# ===============================================

set -e

echo "=========================================="
echo "PerX2CT Training Setup for RunPod"
echo "=========================================="

# 1. Navigate to the project directory (try both possible names)
if [ -d "/workspace/PER-X_2_CT" ]; then
    cd /workspace/PER-X_2_CT
elif [ -d "/workspace/PerX2CT" ]; then
    cd /workspace/PerX2CT
else
    echo "Error: Project directory not found. Running from current directory."
fi

echo "Working directory: $(pwd)"

# 2. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements_flexible.txt

# 5. Check if custom dataset exists
DATA_DIR="/workspace/drr_patient_data"
if [ ! -d "$DATA_DIR" ]; then
    echo "WARNING: Custom dataset directory not found at $DATA_DIR"
    echo "Please ensure your data is placed at $DATA_DIR"
    exit 1
fi

echo "Found custom dataset directory: $DATA_DIR"
echo ""

# 6. List contents of data directory
echo "Contents of $DATA_DIR:"
ls -la "$DATA_DIR" | head -20
echo ""

# 7. Check if data needs preprocessing (NIfTI to H5 conversion)
PROCESSED_DIR="$DATA_DIR/processed_ct128_CTSlice"
if [ ! -d "$PROCESSED_DIR" ]; then
    echo "Preprocessing data (converting NIfTI to H5 slices)..."
    python preprocess_custom_data.py --data_dir "$DATA_DIR" --ct_size 128
    DATA_DIR="$PROCESSED_DIR"
else
    echo "Found preprocessed data at $PROCESSED_DIR"
    DATA_DIR="$PROCESSED_DIR"
fi
echo ""

# 8. Generate dataset list files
echo "Generating dataset list files..."
python generate_custom_dataset_list.py \
    --data_dir "$DATA_DIR" \
    --output_dir ./dataset_list/custom \
    --val_split 0.1

# 8. Verify dataset list files were created
if [ -f "./dataset_list/custom/train.txt" ] && [ -f "./dataset_list/custom/val.txt" ]; then
    echo ""
    echo "Dataset list files created successfully!"
    echo "Train samples: $(wc -l < ./dataset_list/custom/train.txt)"
    echo "Val samples: $(wc -l < ./dataset_list/custom/val.txt)"
else
    echo "ERROR: Failed to create dataset list files"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  python main.py --train True --gpus 0, --name custom_experiment --base configs/PerX2CT_custom.yaml"
echo ""
echo "Training options:"
echo "  --gpus 0,           : Use GPU 0 (adjust for multi-GPU)"
echo "  --name <name>       : Experiment name"
echo "  --base <config>     : Path to config file"
echo "  -r <checkpoint>     : Resume from checkpoint"
echo ""
