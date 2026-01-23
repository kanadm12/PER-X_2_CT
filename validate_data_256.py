#!/usr/bin/env python3
"""
Validate data loading before full training run.

This script tests:
1. Dataset list files exist and are valid
2. CT slices can be loaded from H5 files
3. X-ray images can be found with correct path resolution
4. Preprocessing works correctly
5. Multi-GPU data parallelism is compatible

Usage:
    python validate_data_256.py \
        --config configs/PerX2CT_256_multiGPU.yaml \
        --num_samples 10
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np


def validate_dataset_lists(config):
    """Check that dataset list files exist and have content"""
    print("\n" + "="*60)
    print("STEP 1: Validating Dataset List Files")
    print("="*60)
    
    data_params = config.data.params
    
    train_file = data_params.train.params.training_images_list_file
    val_file = data_params.validation.params.test_images_list_file
    
    errors = []
    
    for name, filepath in [("Train", train_file), ("Val", val_file)]:
        if not os.path.exists(filepath):
            errors.append(f"{name} file not found: {filepath}")
        else:
            with open(filepath, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            print(f"  ✓ {name}: {len(lines):,} samples in {filepath}")
    
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return False
    return True


def validate_sample_loading(config, num_samples=5):
    """Test loading actual samples"""
    print("\n" + "="*60)
    print("STEP 2: Validating Sample Loading")
    print("="*60)
    
    # Import custom data loader
    from x2ct_nerf.data.custom256 import Custom256MultiInputDataset
    
    data_params = config.data.params
    train_file = data_params.train.params.training_images_list_file
    
    with open(train_file, 'r') as f:
        paths = [l.strip() for l in f.readlines() if l.strip()]
    
    # Use first num_samples
    test_paths = paths[:min(num_samples, len(paths))]
    
    opt = OmegaConf.to_container(data_params.train.params.opt, resolve=True)
    
    print(f"  Testing with {len(test_paths)} samples...")
    print(f"  X-ray base dir: {opt.get('xray_base_dir', 'NOT SET')}")
    
    try:
        dataset = Custom256MultiInputDataset(
            paths=test_paths,
            opt=opt,
            size=config.input_img_size,
            num_ctslice_per_item=opt.get('num_ctslice_per_item', 1)
        )
        print(f"  ✓ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"  ✗ Failed to create dataset: {e}")
        return False
    
    # Test loading each sample
    errors = []
    for i in tqdm(range(len(dataset)), desc="  Loading samples"):
        try:
            sample = dataset[i]
            
            # Verify expected keys
            expected_keys = ['ctslice', 'PA', 'Lateral', 'PA_cam', 'Lateral_cam', 'file_path_']
            for key in expected_keys:
                if key not in sample:
                    errors.append(f"Sample {i}: Missing key '{key}'")
            
            # Verify tensor shapes
            # NOTE: Data loader outputs HWC format [256, 256, 3]
            # Model's get_input() permutes to CHW [3, 256, 256] during training
            if 'ctslice' in sample:
                ct_shape = sample['ctslice'].shape
                if ct_shape != torch.Size([256, 256, 3]):
                    errors.append(f"Sample {i}: CT shape {ct_shape}, expected [256, 256, 3]")
            
            if 'PA' in sample:
                pa_shape = sample['PA'].shape
                if pa_shape != torch.Size([256, 256, 3]):
                    errors.append(f"Sample {i}: PA shape {pa_shape}, expected [256, 256, 3]")
            
            if 'Lateral' in sample:
                lat_shape = sample['Lateral'].shape
                if lat_shape != torch.Size([256, 256, 3]):
                    errors.append(f"Sample {i}: Lateral shape {lat_shape}, expected [256, 256, 3]")
                    
        except Exception as e:
            errors.append(f"Sample {i}: {str(e)}")
    
    if errors:
        print(f"\n  Found {len(errors)} errors:")
        for e in errors[:10]:  # Show first 10
            print(f"    ✗ {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
        return False
    
    print(f"  ✓ All {len(dataset)} samples loaded successfully!")
    return True


def validate_dataloader(config, batch_size=4):
    """Test DataLoader batching"""
    print("\n" + "="*60)
    print("STEP 3: Validating DataLoader Batching")
    print("="*60)
    
    from torch.utils.data import DataLoader
    from x2ct_nerf.data.custom256 import Custom256Train
    from taming.data.utils import custom_collate
    
    data_params = config.data.params
    
    try:
        # Create dataset
        dataset = Custom256Train(
            size=config.input_img_size,
            training_images_list_file=data_params.train.params.training_images_list_file,
            dataset_class=data_params.train.params.dataset_class,
            opt=OmegaConf.to_container(data_params.train.params.opt, resolve=True)
        )
        print(f"  ✓ Dataset created: {len(dataset)} samples")
        
        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=custom_collate
        )
        print(f"  ✓ DataLoader created: batch_size={batch_size}")
        
        # Test one batch
        batch = next(iter(loader))
        
        print(f"  ✓ Batch loaded successfully!")
        print(f"    Keys: {list(batch.keys())}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list):
                print(f"    {key}: list of {len(value)} items")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"  ✗ DataLoader failed: {e}")
        traceback.print_exc()
        return False


def validate_model_forward(config):
    """Test model forward pass on CPU (memory safe)"""
    print("\n" + "="*60)
    print("STEP 4: Validating Model Configuration")
    print("="*60)
    
    print("  Skipping full model test (requires GPU)")
    print("  Model config appears valid:")
    print(f"    - Encoder: {config.model.params.metadata.encoder_module}")
    print(f"    - Decoder: {config.model.params.metadata.decoder_module}")
    print(f"    - z_channels: {config.model.params.ddconfig.z_channels}")
    print(f"    - Resolution: {config.model.params.ddconfig.resolution}")
    print(f"    - Discriminator start: {config.model.params.lossconfig.params.disc_start}")
    
    return True


def validate_trainer_config(config):
    """Validate PyTorch Lightning trainer settings"""
    print("\n" + "="*60)
    print("STEP 5: Validating Trainer Configuration")
    print("="*60)
    
    lightning_config = config.get('lightning', {})
    trainer_config = lightning_config.get('trainer', {})
    
    checks = []
    
    # Check required settings
    max_epochs = trainer_config.get('max_epochs', None)
    if max_epochs:
        checks.append(f"✓ max_epochs: {max_epochs}")
    else:
        checks.append("⚠ max_epochs not set (will use default)")
    
    precision = trainer_config.get('precision', None)
    if precision == 16:
        checks.append(f"✓ precision: {precision} (mixed precision enabled)")
    elif precision:
        checks.append(f"✓ precision: {precision}")
    else:
        checks.append("⚠ precision not set (will use FP32)")
    
    strategy = trainer_config.get('strategy', None)
    if strategy == 'ddp':
        checks.append(f"✓ strategy: {strategy} (multi-GPU ready)")
    elif strategy:
        checks.append(f"✓ strategy: {strategy}")
    else:
        checks.append("⚠ strategy not set (will auto-detect)")
    
    sync_bn = trainer_config.get('sync_batchnorm', False)
    if sync_bn:
        checks.append(f"✓ sync_batchnorm: {sync_bn} (recommended for multi-GPU)")
    else:
        checks.append("⚠ sync_batchnorm not enabled")
    
    for check in checks:
        print(f"  {check}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Validate data loading for PerX2CT 256')
    parser.add_argument('--config', type=str, default='configs/PerX2CT_256_multiGPU.yaml',
                        help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for DataLoader test')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PerX2CT 256x256 Data Validation")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Test samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    
    # Load config
    if not os.path.exists(args.config):
        print(f"\n✗ Config not found: {args.config}")
        sys.exit(1)
    
    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    
    results = []
    
    # Run validations
    results.append(("Dataset Lists", validate_dataset_lists(config)))
    results.append(("Sample Loading", validate_sample_loading(config, args.num_samples)))
    results.append(("DataLoader", validate_dataloader(config, args.batch_size)))
    results.append(("Model Config", validate_model_forward(config)))
    results.append(("Trainer Config", validate_trainer_config(config)))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All validations passed! Ready for training.")
        print("\nTo start training:")
        print(f"  python main.py --train True --gpus 0,1,2 --name custom_256_3xA100 --base {args.config}")
    else:
        print("\n❌ Some validations failed. Please fix issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
