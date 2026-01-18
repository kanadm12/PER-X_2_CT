# Training PerX2CT with Custom Dataset on RunPod

This guide explains how to train PerX2CT using your custom DRR patient data located at `/workspace/drr_patient_data/`.

## Prerequisites

1. Your custom dataset should be placed at `/workspace/drr_patient_data/`
2. GPU with at least 16GB VRAM (24GB+ recommended for batch_size=20)

## Expected Data Structure

Your data at `/workspace/drr_patient_data/` should follow this structure:

```
/workspace/drr_patient_data/
├── <dataset_name>_CTSlice/          # e.g., LIDC-HDF5-256_ct128_CTSlice
│   ├── <patient_id>/
│   │   └── ct/
│   │       ├── axial_000.h5
│   │       ├── axial_001.h5
│   │       ├── ...
│   │       ├── coronal_000.h5
│   │       ├── ...
│   │       ├── sagittal_000.h5
│   │       └── ...
│   └── ...
└── <dataset_name>_plastimatch_xray/ # e.g., LIDC-HDF5-256_ct128_plastimatch_xray
    ├── <patient_id>_xray1.png       # PA view
    ├── <patient_id>_xray2.png       # Lateral view
    └── ...
```

### H5 File Format
Each `.h5` file should contain a CT slice with key `'ct'`:
- Shape: 128 x 128 (or your configured resolution)
- Data type: float32 or int16
- Value range: 0-2500 HU (Hounsfield Units)

### X-ray Image Format
- PNG format, 128 x 128 pixels (or your configured resolution)
- Grayscale
- Value range: 0-255

## Quick Start

### Option 1: Using the Setup Script

```bash
cd /workspace/PerX2CT
chmod +x setup_training_runpod.sh
./setup_training_runpod.sh
```

### Option 2: Manual Setup

1. **Set up the environment:**
```bash
cd /workspace/PerX2CT
conda create -n perx2ct python=3.8 -y
conda activate perx2ct
pip install --upgrade pip
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirement.txt
```

2. **Generate dataset list files:**
```bash
python generate_custom_dataset_list.py \
    --data_dir /workspace/drr_patient_data \
    --output_dir ./dataset_list/custom \
    --val_split 0.1
```

3. **Start training:**
```bash
python main.py --train True --gpus 0, --name custom_experiment --base configs/PerX2CT_custom.yaml
```

## Training Configuration

The custom configuration file is at `configs/PerX2CT_custom.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_img_size` | 128 | Input image resolution |
| `input_ct_res` | 128 | CT slice resolution |
| `data.params.batch_size` | 20 | Batch size (reduce if OOM) |
| `data.params.num_workers` | 8 | Data loading workers |
| `model.base_learning_rate` | 4.5e-06 | Base learning rate |

### Adjusting for GPU Memory

If you encounter out-of-memory errors, modify `configs/PerX2CT_custom.yaml`:

```yaml
data:
  params:
    batch_size: 8  # Reduce from 20 to 8 or less
    num_workers: 4  # Reduce if RAM is limited
```

## Training Commands

### Basic Training
```bash
python main.py --train True --gpus 0, --name custom_experiment --base configs/PerX2CT_custom.yaml
```

### Multi-GPU Training
```bash
python main.py --train True --gpus 0,1, --name custom_experiment --base configs/PerX2CT_custom.yaml
```

### Resume Training
```bash
python main.py --train True --gpus 0, --resume logs/custom_experiment/checkpoints/last.ckpt --base configs/PerX2CT_custom.yaml
```

## Monitoring Training

Training logs and checkpoints are saved to `./logs/<experiment_name>/`.

```
logs/custom_experiment/
├── checkpoints/
│   ├── last.ckpt
│   └── epoch=X-step=Y.ckpt
├── configs/
│   └── PerX2CT_custom.yaml
└── testtube/
    └── version_0/
        └── metrics.csv
```

## Evaluation

After training, evaluate your model:

```bash
# Evaluate on validation set
python main_test.py --ckpt_path logs/custom_experiment/checkpoints/last.ckpt \
    --save_dir ./evaluation_results \
    --config_path configs/PerX2CT_custom.yaml \
    --val_test val

# Evaluate on test set
python main_test.py --ckpt_path logs/custom_experiment/checkpoints/last.ckpt \
    --save_dir ./evaluation_results \
    --config_path configs/PerX2CT_custom.yaml \
    --val_test test
```

## Troubleshooting

### "No valid CT slices found"
- Check that your H5 files are in the correct location
- Ensure the folder structure matches the expected format
- Verify X-ray images exist with matching patient IDs

### Out of Memory (OOM)
- Reduce `batch_size` in the config file
- Reduce `num_workers`
- Use gradient accumulation (modify training code)

### Missing X-ray images
- Ensure X-ray images are named `<patient_id>_xray1.png` and `<patient_id>_xray2.png`
- Check that patient IDs match between CT and X-ray folders

### CUDA errors
- Verify CUDA toolkit version matches PyTorch requirements
- Check GPU driver is up to date
- Try reducing batch size

## Contact

For issues with the PerX2CT model, refer to the original repository or contact the authors.
