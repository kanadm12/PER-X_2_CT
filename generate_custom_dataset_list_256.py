#!/usr/bin/env python3
"""
Generate dataset list files for 256x256 custom PerX2CT training.

Creates train.txt, val.txt, test.txt with 80/10/10 split.

Usage:
    python generate_custom_dataset_list_256.py \
        --ct_dir /workspace/processed_ct256_CTSlice \
        --xray_dir /workspace/drr_patient_data_256 \
        --output_dir ./dataset_list/custom256
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import random


def main():
    parser = argparse.ArgumentParser(description='Generate dataset lists for 256x256 training')
    parser.add_argument('--ct_dir', type=str, required=True,
                        help='Directory containing processed CT slices')
    parser.add_argument('--xray_dir', type=str, required=True,
                        help='Directory containing X-ray images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for dataset list files')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    ct_dir = Path(args.ct_dir)
    xray_dir = Path(args.xray_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(args.seed)
    
    print("="*60)
    print("Generating 256x256 Dataset Lists")
    print("="*60)
    print(f"CT directory:   {ct_dir}")
    print(f"X-ray directory: {xray_dir}")
    print(f"Output:         {output_dir}")
    print("="*60)
    
    # Find all patient directories in CT folder
    patient_dirs = sorted([d for d in ct_dir.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patients in CT directory")
    
    # Build X-ray index for fast lookup
    print("Building X-ray index...")
    xray_index = {}
    for patient_folder in tqdm(xray_dir.iterdir(), desc="Indexing X-rays"):
        if patient_folder.is_dir():
            patient_id = patient_folder.name
            pa_files = list(patient_folder.glob("*_pa_drr*.png"))
            lat_files = list(patient_folder.glob("*_lat_drr*.png"))
            if pa_files and lat_files:
                xray_index[patient_id] = True
    
    print(f"Found {len(xray_index)} patients with valid X-rays")
    
    # Collect valid samples
    print("Collecting valid samples...")
    all_samples = []
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir.name
        
        # Check if patient has X-rays
        if patient_id not in xray_index:
            continue
        
        # Find all H5 slices for this patient
        h5_files = sorted(patient_dir.glob("*.h5"))
        
        for h5_file in h5_files:
            # Format: /path/to/ct/patient_id/patient_id_000.h5
            all_samples.append(str(h5_file))
    
    print(f"Total valid samples: {len(all_samples):,}")
    
    # Get unique patients for splitting
    patients = list(set([Path(s).parent.name for s in all_samples]))
    random.shuffle(patients)
    
    n_train = int(len(patients) * args.train_ratio)
    n_val = int(len(patients) * args.val_ratio)
    
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])
    
    # Split samples
    train_samples = [s for s in all_samples if Path(s).parent.name in train_patients]
    val_samples = [s for s in all_samples if Path(s).parent.name in val_patients]
    test_samples = [s for s in all_samples if Path(s).parent.name in test_patients]
    
    # Write files
    for name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        output_path = output_dir / f"{name}.txt"
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(sample + '\n')
        print(f"Wrote {len(samples):,} samples to {output_path.name}")
    
    # Summary
    print("\n" + "="*60)
    print("DATASET SPLIT SUMMARY (256x256)")
    print("="*60)
    print(f"Train: {len(train_samples):,} samples ({len(train_patients)} patients)")
    print(f"Val:   {len(val_samples):,} samples ({len(val_patients)} patients)")
    print(f"Test:  {len(test_samples):,} samples ({len(test_patients)} patients)")
    print(f"Total: {len(all_samples):,} samples ({len(patients)} patients)")
    print("="*60)


if __name__ == "__main__":
    main()
