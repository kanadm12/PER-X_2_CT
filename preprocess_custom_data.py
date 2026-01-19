#!/usr/bin/env python3
"""
Preprocessing script to convert custom DRR patient data to PerX2CT format.

Input structure (your data):
    /workspace/drr_patient_data/
    └── <patient_id>/
        ├── <patient_id>.nii.gz           # CT volume
        ├── <patient_id>_pa_drr_flipped.png   # PA X-ray
        └── <patient_id>_lat_drr_flipped.png  # Lateral X-ray

Output structure (PerX2CT format):
    /workspace/drr_patient_data/
    ├── processed_ct128_CTSlice/
    │   └── <patient_id>/
    │       └── ct/
    │           ├── axial_000.h5
    │           ├── axial_001.h5
    │           └── ...
    └── processed_ct128_plastimatch_xray/
        ├── <patient_id>_xray1.png   # PA
        └── <patient_id>_xray2.png   # Lateral
"""

import os
import glob
import shutil
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    print("Installing nibabel...")
    os.system("pip install nibabel")
    import nibabel as nib

try:
    from PIL import Image
except ImportError:
    print("Installing Pillow...")
    os.system("pip install Pillow")
    from PIL import Image

try:
    from scipy.ndimage import zoom
except ImportError:
    print("Installing scipy...")
    os.system("pip install scipy")
    from scipy.ndimage import zoom


def resize_volume(volume, target_shape=(128, 128, 128)):
    """Resize 3D volume to target shape."""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)


def normalize_ct(volume, min_val=-1000, max_val=2500):
    """Normalize CT values to 0-2500 range (clip and shift)."""
    volume = np.clip(volume, min_val, max_val)
    volume = volume - min_val  # Shift to start at 0
    return volume.astype(np.float32)


def process_patient(patient_dir, output_ct_dir, output_xray_dir, ct_size=128, skip_existing=True):
    """Process a single patient's data."""
    patient_id = os.path.basename(patient_dir)
    
    # Check if already preprocessed (skip if all expected files exist)
    patient_ct_dir = os.path.join(output_ct_dir, patient_id, "ct")
    xray1_path = os.path.join(output_xray_dir, f"{patient_id}_xray1_flipped.png")
    xray2_path = os.path.join(output_xray_dir, f"{patient_id}_xray2_flipped.png")
    last_axial = os.path.join(patient_ct_dir, f"axial_{ct_size-1:03d}.h5")
    
    if skip_existing:
        if os.path.exists(last_axial) and os.path.exists(xray1_path) and os.path.exists(xray2_path):
            return "skipped"
    
    # Find NIfTI file
    nii_files = glob.glob(os.path.join(patient_dir, "*.nii.gz"))
    if not nii_files:
        nii_files = glob.glob(os.path.join(patient_dir, "*.nii"))
    
    if not nii_files:
        print(f"  WARNING: No NIfTI file found for {patient_id}")
        return False
    
    nii_path = nii_files[0]
    
    # Find X-ray images
    pa_files = glob.glob(os.path.join(patient_dir, "*_pa_drr*.png"))
    lat_files = glob.glob(os.path.join(patient_dir, "*_lat_drr*.png"))
    
    if not pa_files or not lat_files:
        print(f"  WARNING: Missing X-ray images for {patient_id}")
        return False
    
    pa_path = pa_files[0]
    lat_path = lat_files[0]
    
    # Create output directories
    os.makedirs(patient_ct_dir, exist_ok=True)
    os.makedirs(output_xray_dir, exist_ok=True)
    
    # Load and process CT volume
    try:
        nii = nib.load(nii_path)
        ct_volume = nii.get_fdata()
        
        # Normalize CT values
        ct_volume = normalize_ct(ct_volume)
        
        # Resize to target resolution
        if ct_volume.shape != (ct_size, ct_size, ct_size):
            ct_volume = resize_volume(ct_volume, (ct_size, ct_size, ct_size))
        
        # Save axial slices (z-axis)
        for i in range(ct_size):
            slice_data = ct_volume[:, :, i]
            h5_path = os.path.join(patient_ct_dir, f"axial_{i:03d}.h5")
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('ct', data=slice_data.astype(np.float32))
        
        # Save coronal slices (y-axis)
        for i in range(ct_size):
            slice_data = ct_volume[:, i, :]
            h5_path = os.path.join(patient_ct_dir, f"coronal_{i:03d}.h5")
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('ct', data=slice_data.astype(np.float32))
        
        # Save sagittal slices (x-axis)
        for i in range(ct_size):
            slice_data = ct_volume[i, :, :]
            h5_path = os.path.join(patient_ct_dir, f"sagittal_{i:03d}.h5")
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('ct', data=slice_data.astype(np.float32))
        
    except Exception as e:
        print(f"  ERROR processing CT for {patient_id}: {e}")
        return False
    
    # Copy and rename X-ray images
    try:
        # Resize X-rays to 128x128 if needed
        pa_img = Image.open(pa_path).convert('L')  # Grayscale
        lat_img = Image.open(lat_path).convert('L')
        
        if pa_img.size != (ct_size, ct_size):
            pa_img = pa_img.resize((ct_size, ct_size), Image.BILINEAR)
        if lat_img.size != (ct_size, ct_size):
            lat_img = lat_img.resize((ct_size, ct_size), Image.BILINEAR)
        
        # Save with expected naming convention
        pa_img.save(os.path.join(output_xray_dir, f"{patient_id}_xray1.png"))
        lat_img.save(os.path.join(output_xray_dir, f"{patient_id}_xray2.png"))
        
    except Exception as e:
        print(f"  ERROR processing X-rays for {patient_id}: {e}")
        return False
    
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert custom DRR data to PerX2CT format")
    parser.add_argument("--data_dir", type=str, default="/workspace/drr_patient_data",
                        help="Path to raw patient data")
    parser.add_argument("--ct_size", type=int, default=128,
                        help="Target CT resolution (default: 128)")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing of already processed patients")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    ct_size = args.ct_size
    skip_existing = not args.force
    
    # Output directories
    output_ct_dir = data_dir / f"processed_ct{ct_size}_CTSlice"
    output_xray_dir = data_dir / f"processed_ct{ct_size}_plastimatch_xray"
    
    print("=" * 60)
    print("Converting Custom DRR Data to PerX2CT Format")
    print("=" * 60)
    print(f"Input directory: {data_dir}")
    print(f"CT output: {output_ct_dir}")
    print(f"X-ray output: {output_xray_dir}")
    print(f"Target CT resolution: {ct_size}x{ct_size}x{ct_size}")
    print(f"Skip existing: {skip_existing}")
    print()
    
    # Find all patient directories
    patient_dirs = [d for d in data_dir.iterdir() 
                    if d.is_dir() and not d.name.startswith('processed_')]
    
    if not patient_dirs:
        print("ERROR: No patient directories found!")
        return
    
    print(f"Found {len(patient_dirs)} patient directories")
    print()
    
    success = 0
    failed = 0
    skipped = 0
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        result = process_patient(str(patient_dir), str(output_ct_dir), str(output_xray_dir), ct_size, skip_existing)
        if result == "skipped":
            skipped += 1
        elif result:
            success += 1
        else:
            failed += 1
    
    print()
    print("=" * 60)
    print("Processing Complete!")
    print(f"Newly processed: {success} patients")
    print(f"Skipped (already done): {skipped} patients")
    print(f"Failed: {failed} patients")
    print()
    print("Next steps:")
    print("  1. Run: python generate_custom_dataset_list.py --data_dir /workspace/drr_patient_data/processed_ct128_CTSlice")
    print("  2. Run: python main.py --train True --gpus 0, --name custom --base configs/PerX2CT_custom.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
