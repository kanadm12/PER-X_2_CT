#!/usr/bin/env python3
"""
Script to generate training and validation dataset list files from custom DRR patient data.
This script scans your custom dataset directory and creates train.txt and val.txt files
that are compatible with the PerX2CT training pipeline.

Expected data structure at /workspace/drr_patient_data/:
    /workspace/drr_patient_data/
    ├── <patient_id>/
    │   ├── ct/
    │   │   ├── axial_000.h5
    │   │   ├── axial_001.h5
    │   │   ├── ...
    │   │   ├── coronal_000.h5
    │   │   ├── ...
    │   │   ├── sagittal_000.h5
    │   │   └── ...
    │   └── xray/
    │       ├── <patient_id>_xray1.png  (PA view)
    │       └── <patient_id>_xray2.png  (Lateral view)
    └── ...

OR alternative structure:
    /workspace/drr_patient_data/
    ├── ct_slices/  (or LIDC-HDF5-256_ct128_CTSlice/)
    │   └── <patient_id>/
    │       └── ct/
    │           ├── axial_000.h5
    │           └── ...
    └── xrays/  (or LIDC-HDF5-256_ct128_plastimatch_xray/)
        ├── <patient_id>_xray1.png
        └── <patient_id>_xray2.png

Usage:
    python generate_custom_dataset_list.py --data_dir /workspace/drr_patient_data --output_dir ./dataset_list/custom --val_split 0.1
"""

import os
import glob
import argparse
import random
from pathlib import Path


def find_ct_slices_and_xrays(data_dir):
    """
    Find all CT slices and their corresponding X-ray images.
    Returns a list of valid CT slice paths that have matching X-rays.
    """
    ct_slices = []
    
    # Find the processed CT directory
    data_path = Path(data_dir)
    
    # Look for H5 files in various patterns
    patterns = [
        # Pattern 1: processed_ct128_CTSlice/<patient>/ct/*.h5
        data_path / "*_CTSlice" / "*" / "ct" / "*.h5",
        # Pattern 2: Direct <patient>/ct/*.h5
        data_path / "*" / "ct" / "*.h5",
        # Pattern 3: Given path is already the CTSlice folder
        data_path / "*" / "ct" / "*.h5",
    ]
    
    found_files = []
    for pattern in patterns:
        files = list(data_path.glob(str(pattern).replace(str(data_path) + "/", "")))
        if files:
            found_files.extend(files)
            print(f"Found {len(files)} CT slice files")
            break
    
    if not found_files:
        # Try direct glob
        found_files = list(data_path.rglob("*.h5"))
        if found_files:
            print(f"Found {len(found_files)} H5 files via recursive search")
    
    if not found_files:
        print(f"WARNING: No CT slice files (.h5) found in {data_dir}")
        print("Please check your data directory structure.")
        print("Did you run preprocess_custom_data.py first?")
        return []
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    # Find the X-ray directory (sibling to CTSlice folder)
    xray_dir = None
    for f in found_files:
        path_str = str(f)
        if "_CTSlice" in path_str:
            xray_dir = path_str.split("_CTSlice")[0] + "_plastimatch_xray"
            break
    
    if xray_dir is None:
        # Try to find xray folder as sibling
        parent = data_path.parent if data_path.name.endswith("_CTSlice") else data_path
        xray_candidates = list(parent.glob("*_xray")) + list(parent.glob("*_plastimatch_xray"))
        if xray_candidates:
            xray_dir = str(xray_candidates[0])
    
    print(f"Looking for X-rays in: {xray_dir}")
    
    # Validate that corresponding X-ray images exist
    valid_slices = []
    missing_xrays = set()
    
    for ct_path in found_files:
        ct_path_str = str(ct_path).replace("\\", "/")
        
        # Extract patient ID from path
        path_parts = ct_path_str.split("/")
        patient_id = None
        for i, part in enumerate(path_parts):
            if part == "ct" and i > 0:
                patient_id = path_parts[i-1]
                break
        
        if patient_id and xray_dir:
            xray1 = os.path.join(xray_dir, f"{patient_id}_xray1.png")
            xray2 = os.path.join(xray_dir, f"{patient_id}_xray2.png")
            
            if os.path.exists(xray1) and os.path.exists(xray2):
                valid_slices.append(ct_path_str)
            else:
                missing_xrays.add(patient_id)
        elif patient_id:
            # No xray_dir found, include anyway
            valid_slices.append(ct_path_str)
    
    if missing_xrays:
        print(f"\nWARNING: {len(missing_xrays)} patients have missing X-ray images:")
        for pid in list(missing_xrays)[:5]:
            print(f"  - {pid}")
        if len(missing_xrays) > 5:
            print(f"  ... and {len(missing_xrays) - 5} more")
    
    print(f"\nFound {len(valid_slices)} valid CT slices with corresponding X-rays")
    return valid_slices


def create_dataset_lists(ct_slices, output_dir, val_split=0.1, seed=42):
    """
    Split CT slices into train and validation sets and save to text files.
    """
    if not ct_slices:
        print("ERROR: No valid CT slices found. Cannot create dataset lists.")
        return
    
    random.seed(seed)
    
    # Group by patient to avoid data leakage
    patient_slices = {}
    for ct_path in ct_slices:
        path_parts = ct_path.replace("\\", "/").split("/")
        for i, part in enumerate(path_parts):
            if part == "ct" and i > 0:
                patient_id = path_parts[i-1]
                if patient_id not in patient_slices:
                    patient_slices[patient_id] = []
                patient_slices[patient_id].append(ct_path)
                break
    
    # Split patients into train/val
    patients = list(patient_slices.keys())
    random.shuffle(patients)
    
    n_val = max(1, int(len(patients) * val_split))
    val_patients = set(patients[:n_val])
    train_patients = set(patients[n_val:])
    
    train_slices = []
    val_slices = []
    
    for patient, slices in patient_slices.items():
        if patient in val_patients:
            val_slices.extend(slices)
        else:
            train_slices.extend(slices)
    
    # Shuffle the slices
    random.shuffle(train_slices)
    random.shuffle(val_slices)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train.txt
    train_file = os.path.join(output_dir, "train.txt")
    with open(train_file, "w") as f:
        for path in train_slices:
            # Convert to relative path starting with ./data/
            rel_path = convert_to_relative_path(path)
            f.write(f"{rel_path}\n")
    print(f"Created {train_file} with {len(train_slices)} samples from {len(train_patients)} patients")
    
    # Save val.txt
    val_file = os.path.join(output_dir, "val.txt")
    with open(val_file, "w") as f:
        for path in val_slices:
            rel_path = convert_to_relative_path(path)
            f.write(f"{rel_path}\n")
    print(f"Created {val_file} with {len(val_slices)} samples from {len(val_patients)} patients")
    
    # Also create test.txt (same as val for now)
    test_file = os.path.join(output_dir, "test.txt")
    with open(test_file, "w") as f:
        for path in val_slices:
            rel_path = convert_to_relative_path(path)
            f.write(f"{rel_path}\n")
    print(f"Created {test_file} with {len(val_slices)} samples")


def convert_to_relative_path(abs_path):
    """
    Convert absolute path to relative path format expected by the dataset loader.
    The loader expects paths like: ./data/<folder>/<patient_id>/ct/<slice>.h5
    """
    abs_path = abs_path.replace("\\", "/")
    
    # If path already contains 'data/' or starts with './', use as is
    if "/data/" in abs_path:
        idx = abs_path.find("/data/")
        return "." + abs_path[idx:]
    elif abs_path.startswith("./"):
        return abs_path
    
    # Try to find the dataset root
    # Expected: .../<dataset_name>/<patient_id>/ct/<slice>.h5
    parts = abs_path.split("/")
    
    # Find 'ct' folder position
    for i, part in enumerate(parts):
        if part == "ct" and i >= 2:
            # Take dataset folder, patient folder, ct folder, and filename
            rel_parts = parts[i-2:]
            return "./data/" + "/".join(rel_parts)
    
    # Fallback: just use the path as is with ./data/ prefix
    return "./data/" + os.path.basename(os.path.dirname(os.path.dirname(abs_path))) + "/" + \
           os.path.basename(os.path.dirname(abs_path)) + "/" + os.path.basename(abs_path)


def main():
    parser = argparse.ArgumentParser(description="Generate dataset list files for PerX2CT training")
    parser.add_argument("--data_dir", type=str, default="/workspace/drr_patient_data",
                        help="Path to the custom dataset directory")
    parser.add_argument("--output_dir", type=str, default="./dataset_list/custom",
                        help="Output directory for train.txt and val.txt")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of patients to use for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Scanning for CT slices in: {args.data_dir}")
    print("-" * 50)
    
    # Find all valid CT slices
    ct_slices = find_ct_slices_and_xrays(args.data_dir)
    
    if ct_slices:
        print("-" * 50)
        create_dataset_lists(ct_slices, args.output_dir, args.val_split, args.seed)
        print("-" * 50)
        print("\nDataset list generation complete!")
        print(f"\nTo start training, run:")
        print(f"  python main.py --train True --gpus 0, --name custom_experiment --base configs/PerX2CT_custom.yaml")
    else:
        print("\nNo valid data found. Please check your data directory structure.")
        print("\nExpected structure:")
        print("  /workspace/drr_patient_data/")
        print("  ├── <dataset_name>_CTSlice/")
        print("  │   └── <patient_id>/")
        print("  │       └── ct/")
        print("  │           ├── axial_000.h5")
        print("  │           ├── coronal_000.h5")
        print("  │           └── sagittal_000.h5")
        print("  └── <dataset_name>_plastimatch_xray/")
        print("      ├── <patient_id>_xray1.png")
        print("      └── <patient_id>_xray2.png")


if __name__ == "__main__":
    main()
