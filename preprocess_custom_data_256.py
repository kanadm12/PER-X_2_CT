#!/usr/bin/env python3
"""
Preprocess custom DRR patient data for PerX2CT training at 256x256 resolution.

Input structure:
    /workspace/drr_patient_data/
        patient_001/
            patient_001.nii.gz (CT volume)
            patient_001_pa_drr_flipped.png (PA X-ray)
            patient_001_lat_drr_flipped.png (Lateral X-ray)
        patient_002/
            ...

Output structure:
    /workspace/processed_ct256_CTSlice/
        patient_001/
            patient_001_000.h5 (slice 0)
            patient_001_001.h5 (slice 1)
            ...
    /workspace/drr_patient_data/ (X-rays stay in place, resized to 256x256)

Usage:
    python preprocess_custom_data_256.py \
        --input_dir /workspace/drr_patient_data \
        --output_dir /workspace/processed_ct256_CTSlice \
        --xray_output_dir /workspace/drr_patient_data_256 \
        --force
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import h5py
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Constants for 256x256 resolution
CT_SIZE = 256
XRAY_SIZE = 256
NUM_SLICES = 256  # More slices for higher resolution


def resize_volume_to_256(volume):
    """Resize 3D volume to 256x256x256"""
    from scipy.ndimage import zoom as scipy_zoom
    
    current_shape = volume.shape
    target_shape = (CT_SIZE, CT_SIZE, NUM_SLICES)
    
    zoom_factors = (
        target_shape[0] / current_shape[0],
        target_shape[1] / current_shape[1],
        target_shape[2] / current_shape[2]
    )
    
    resized = scipy_zoom(volume, zoom_factors, order=1, mode='nearest')
    return resized


def process_ct_volume(nifti_path, output_dir, patient_id, force=False):
    """Process a single CT volume into 256x256 H5 slices"""
    
    output_patient_dir = Path(output_dir) / patient_id
    
    # Check if already processed
    if not force and output_patient_dir.exists():
        existing_files = list(output_patient_dir.glob("*.h5"))
        if len(existing_files) >= NUM_SLICES:
            return patient_id, "skipped", len(existing_files)
    
    output_patient_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load NIfTI
        nii = nib.load(nifti_path)
        volume = nii.get_fdata()
        
        # Resize to 256x256x256
        volume = resize_volume_to_256(volume)
        
        # Ensure correct orientation (H, W, D)
        if volume.shape[0] != CT_SIZE or volume.shape[1] != CT_SIZE:
            volume = np.transpose(volume, (1, 0, 2))
        
        # Process each slice
        slices_saved = 0
        for slice_idx in range(NUM_SLICES):
            ct_slice = volume[:, :, slice_idx]
            
            # Save as H5 with both 'ct' and 'ctslice' keys for compatibility
            h5_path = output_patient_dir / f"{patient_id}_{slice_idx:03d}.h5"
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('ct', data=ct_slice.astype(np.float32), compression='gzip')
                f.create_dataset('ctslice', data=ct_slice.astype(np.float32), compression='gzip')
            
            slices_saved += 1
        
        return patient_id, "success", slices_saved
        
    except Exception as e:
        return patient_id, f"error: {str(e)}", 0


def resize_xray(xray_path, output_path):
    """Resize X-ray to 256x256"""
    try:
        img = Image.open(xray_path).convert('L')
        img_resized = img.resize((XRAY_SIZE, XRAY_SIZE), Image.BILINEAR)
        img_resized.save(output_path)
        return True
    except Exception as e:
        print(f"Error resizing {xray_path}: {e}")
        return False


def process_patient(args):
    """Process a single patient (CT + X-rays)"""
    patient_dir, output_ct_dir, output_xray_dir, force = args
    
    patient_id = patient_dir.name
    
    # Find files
    nifti_files = list(patient_dir.glob("*.nii.gz"))
    if not nifti_files:
        return patient_id, "error: no nifti", 0
    
    nifti_path = nifti_files[0]
    
    # Process CT volume
    result = process_ct_volume(nifti_path, output_ct_dir, patient_id, force)
    
    # Process X-rays (resize to 256x256) - PRIORITIZE _flipped versions
    if output_xray_dir:
        output_patient_xray_dir = Path(output_xray_dir) / patient_id
        output_patient_xray_dir.mkdir(parents=True, exist_ok=True)
        
        # Find flipped DRRs first (preferred)
        flipped_drrs = list(patient_dir.glob("*_drr_flipped.png"))
        
        if flipped_drrs:
            # Use flipped DRRs
            for xray_file in flipped_drrs:
                output_xray_path = output_patient_xray_dir / xray_file.name
                if not output_xray_path.exists() or force:
                    resize_xray(xray_file, output_xray_path)
        else:
            # Fallback to non-flipped DRRs if no flipped versions exist
            for xray_file in patient_dir.glob("*_drr*.png"):
                output_xray_path = output_patient_xray_dir / xray_file.name
                if not output_xray_path.exists() or force:
                    resize_xray(xray_file, output_xray_path)
        
        # Copy original NIfTI for reference
        import shutil
        nifti_dest = output_patient_xray_dir / nifti_path.name
        if not nifti_dest.exists():
            shutil.copy2(nifti_path, nifti_dest)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Preprocess custom data at 256x256 resolution')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing patient folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed CT slices')
    parser.add_argument('--xray_output_dir', type=str, default=None,
                        help='Output directory for resized X-rays (optional)')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing of existing files')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    xray_output_dir = Path(args.xray_output_dir) if args.xray_output_dir else None
    if xray_output_dir:
        xray_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all patient directories
    patient_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    print("="*60)
    print("PerX2CT 256x256 Data Preprocessing")
    print("="*60)
    print(f"Input directory:  {input_dir}")
    print(f"Output CT dir:    {output_dir}")
    print(f"Output X-ray dir: {xray_output_dir or 'N/A'}")
    print(f"Total patients:   {len(patient_dirs)}")
    print(f"Resolution:       {CT_SIZE}x{CT_SIZE}x{NUM_SLICES}")
    print(f"Force reprocess:  {args.force}")
    print("="*60)
    
    # Process patients
    task_args = [(pd, output_dir, xray_output_dir, args.force) for pd in patient_dirs]
    
    results = {"success": 0, "skipped": 0, "error": 0}
    errors = []
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_patient, arg): arg[0].name for arg in task_args}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            patient_id = futures[future]
            try:
                pid, status, num_slices = future.result()
                if status == "success":
                    results["success"] += 1
                elif status == "skipped":
                    results["skipped"] += 1
                else:
                    results["error"] += 1
                    errors.append((pid, status))
            except Exception as e:
                results["error"] += 1
                errors.append((patient_id, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Success:  {results['success']}")
    print(f"Skipped:  {results['skipped']}")
    print(f"Errors:   {results['error']}")
    
    if errors:
        print("\nErrors:")
        for pid, err in errors[:10]:
            print(f"  {pid}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    total_slices = (results['success'] + results['skipped']) * NUM_SLICES
    print(f"\nTotal CT slices: {total_slices:,}")
    print("="*60)


if __name__ == "__main__":
    main()
