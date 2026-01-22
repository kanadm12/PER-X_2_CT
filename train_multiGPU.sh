#!/bin/bash
# =============================================================================
# Multi-GPU Training Script for PerX2CT on 3x A100 80GB Cluster
# =============================================================================
#
# Usage:
#   chmod +x train_multiGPU.sh
#   ./train_multiGPU.sh
#
# Or run components separately - see individual commands below
# =============================================================================

set -e  # Exit on error

# Configuration
NUM_GPUS=3
CONFIG="configs/PerX2CT_256_multiGPU.yaml"
EXPERIMENT_NAME="custom_256_3xA100"

# Directories
DATA_INPUT_DIR="/workspace/drr_patient_data"
CT_OUTPUT_DIR="/workspace/processed_ct256_CTSlice"
XRAY_OUTPUT_DIR="/workspace/drr_patient_data_256"
DATASET_LIST_DIR="./dataset_list/custom256"

echo "============================================================"
echo "PerX2CT Multi-GPU Training Pipeline"
echo "============================================================"
echo "GPUs: ${NUM_GPUS}x A100 80GB"
echo "Config: ${CONFIG}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "============================================================"

# =============================================================================
# STEP 1: Pull latest code
# =============================================================================
echo ""
echo "[Step 1/4] Pulling latest code..."
git pull origin main

# =============================================================================
# STEP 2: Preprocess data at 256x256
# =============================================================================
echo ""
echo "[Step 2/4] Preprocessing data at 256x256 resolution..."

if [ -d "${CT_OUTPUT_DIR}" ] && [ "$(ls -A ${CT_OUTPUT_DIR})" ]; then
    echo "  CT data directory exists and is not empty."
    read -p "  Reprocess all data? (y/N): " reprocess
    if [ "$reprocess" = "y" ] || [ "$reprocess" = "Y" ]; then
        python preprocess_custom_data_256.py \
            --input_dir ${DATA_INPUT_DIR} \
            --output_dir ${CT_OUTPUT_DIR} \
            --xray_output_dir ${XRAY_OUTPUT_DIR} \
            --num_workers 16 \
            --force
    else
        echo "  Skipping preprocessing (using existing data)"
    fi
else
    python preprocess_custom_data_256.py \
        --input_dir ${DATA_INPUT_DIR} \
        --output_dir ${CT_OUTPUT_DIR} \
        --xray_output_dir ${XRAY_OUTPUT_DIR} \
        --num_workers 16
fi

# =============================================================================
# STEP 3: Generate dataset lists
# =============================================================================
echo ""
echo "[Step 3/4] Generating dataset lists..."

if [ -f "${DATASET_LIST_DIR}/train.txt" ]; then
    echo "  Dataset lists already exist."
    read -p "  Regenerate? (y/N): " regen
    if [ "$regen" = "y" ] || [ "$regen" = "Y" ]; then
        python generate_custom_dataset_list_256.py \
            --ct_dir ${CT_OUTPUT_DIR} \
            --xray_dir ${XRAY_OUTPUT_DIR} \
            --output_dir ${DATASET_LIST_DIR}
    else
        echo "  Using existing dataset lists"
    fi
else
    python generate_custom_dataset_list_256.py \
        --ct_dir ${CT_OUTPUT_DIR} \
        --xray_dir ${XRAY_OUTPUT_DIR} \
        --output_dir ${DATASET_LIST_DIR}
fi

# Show dataset stats
echo ""
echo "Dataset Statistics:"
echo "  Train samples: $(wc -l < ${DATASET_LIST_DIR}/train.txt)"
echo "  Val samples:   $(wc -l < ${DATASET_LIST_DIR}/val.txt)"
echo "  Test samples:  $(wc -l < ${DATASET_LIST_DIR}/test.txt)"

# =============================================================================
# STEP 3.5: Validate data loading (catch issues before training)
# =============================================================================
echo ""
echo "[Step 3.5/5] Validating data loading..."
python validate_data_256.py --config ${CONFIG} --num_samples 5 --batch_size 4

if [ $? -ne 0 ]; then
    echo "❌ Data validation failed! Please fix issues before training."
    exit 1
fi

# =============================================================================
# STEP 4: Train with multi-GPU
# =============================================================================
echo ""
echo "[Step 4/5] Starting multi-GPU training..."
echo ""
echo "Training Configuration:"
echo "  GPUs: 0,1,2 (${NUM_GPUS}x A100)"
echo "  Batch size per GPU: 32"
echo "  Total batch size: $((32 * NUM_GPUS))"
echo "  Resolution: 256x256"
echo "  Max epochs: 20"
echo "  Discriminator: Enabled (starts at step 30000)"
echo "  Data loader: Custom256 (fixed X-ray path resolution)"
echo ""

# Start training
python main.py \
    --train True \
    --gpus 0,1,2 \
    --name ${EXPERIMENT_NAME} \
    --base ${CONFIG}

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo "Checkpoints saved to: ./logs/PerX2CT/${EXPERIMENT_NAME}_*/"
echo ""
echo "To run inference:"
echo "  python run_inference_custom.py \\"
echo "    --ckpt_path ./logs/PerX2CT/${EXPERIMENT_NAME}_*/checkpoints/best_psnr.ckpt \\"
echo "    --data_dir ${DATA_INPUT_DIR} \\"
echo "    --output_dir ./inference_outputs"
echo "============================================================"
