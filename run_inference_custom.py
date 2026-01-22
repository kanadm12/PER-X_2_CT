#!/usr/bin/env python3
"""
Simple inference script for custom trained PerX2CT model
Reconstructs 3D CT volume from bidirectional DRR images (PA + Lateral)

Usage:
    python run_inference_custom.py \
        --ckpt_path logs/PerX2CT/custom_experiment__20260118_062224/checkpoints/epoch=000013.ckpt \
        --data_dir /workspace/drr_patient_data \
        --output_dir ./inference_outputs
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from importlib import import_module
import argparse
from tqdm import tqdm
import imageio
import math
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

# Global resolution setting (will be set from command line)
INFERENCE_RESOLUTION = 128


def load_model(config_path, ckpt_path, device='cuda'):
    """Load the trained model from checkpoint"""
    config = OmegaConf.load(config_path)
    
    # Get model class
    model_target = config.model.target
    model_module = model_target.split(".")
    model_module, model_class = model_module[:-1], model_module[-1]
    model_module = ".".join(model_module)
    model_class = getattr(import_module(model_module), model_class)
    
    # Instantiate model
    model = model_class(**config.model.params)
    
    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model, config


def load_and_preprocess_drr(drr_path, target_size=None, apply_camera_transform=True, camera_type='PA'):
    """
    Load and preprocess a DRR image to match training data format.
    
    The training preprocessing does:
    1. Load grayscale image
    2. Apply camera-specific transforms (flip for PA, transpose+flip for Lateral)
    3. Normalize to [0, 1]
    4. Stack to 3 channels
    
    Args:
        target_size: If None, uses global INFERENCE_RESOLUTION (default 128)
    """
    global INFERENCE_RESOLUTION
    if target_size is None:
        target_size = INFERENCE_RESOLUTION if 'INFERENCE_RESOLUTION' in globals() else 128
        
    # Load image
    img = Image.open(drr_path).convert('L')
    
    # Resize if needed
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to numpy
    img_array = np.array(img, dtype=np.float32)
    
    # Apply camera-specific transformations (matching training preprocessing)
    # From base.py: apply_preprocessing_xray_according2cam
    if apply_camera_transform:
        if camera_type == 'PA':
            img_array = np.fliplr(img_array)
        elif camera_type == 'Lateral':
            img_array = np.transpose(img_array, (1, 0))
            img_array = np.flipud(img_array)
    
    # Normalize to [0, 1] (training uses XRAY_MIN_MAX = [0, 255])
    img_array = img_array / 255.0
    
    # Create 3-channel image (H, W, 3) to match training format
    img_array = np.expand_dims(img_array, -1)
    img_array = np.concatenate([img_array, img_array, img_array], axis=-1)
    
    return img_array


def reconstruct_ct_volume(model, pa_tensor, lat_tensor, device, num_slices=128):
    """
    Reconstruct a 3D CT volume by iterating through all axial slice positions.
    
    Args:
        model: Trained PerX2CT model
        pa_tensor: PA DRR tensor (1, H, W, 3)
        lat_tensor: Lateral DRR tensor (1, H, W, 3)
        device: torch device
        num_slices: Number of slices to reconstruct
    
    Returns:
        3D numpy array of shape (H, W, num_slices)
    """
    # Camera poses (same as training)
    pa_cam = torch.tensor([[0.0, 0.0]], device=device)
    lateral_cam = torch.tensor([[math.pi / 2, math.pi / 2]], device=device)
    
    volume_slices = []
    
    for slice_idx in tqdm(range(num_slices), desc="Reconstructing slices", leave=False):
        # Create dummy CT slice (model only needs the shape)
        dummy_ct = torch.zeros_like(pa_tensor)
        
        # Create batch dict matching training format
        batch = {
            'ctslice': dummy_ct,
            'PA': pa_tensor,
            'Lateral': lat_tensor,
            'PA_cam': pa_cam,
            'Lateral_cam': lateral_cam,
            'file_path_': [f"inference/axial_{slice_idx:03d}.h5"],
            'image_key': 'ctslice',
        }
        
        # Process batch through model's get_input
        batch = model.get_input(batch)
        batch['image_key'] = 'ctslice'
        
        # Run forward pass
        with torch.no_grad():
            output_dict, _ = model(batch)
        
        # Extract reconstructed slice
        recon = output_dict['outputs']  # Shape: (1, 3, H, W)
        recon_np = recon[0, 0].cpu().numpy()  # Take first channel: (H, W)
        
        volume_slices.append(recon_np)
    
    # Stack into 3D volume
    volume = np.stack(volume_slices, axis=2)  # (H, W, D)
    
    return volume


def find_patient_files(patient_dir):
    """Find PA and Lateral DRR files in a patient directory"""
    patient_dir = Path(patient_dir)
    
    pa_drr = None
    lat_drr = None
    gt_ct = None
    
    for f in patient_dir.glob("*.png"):
        fname_lower = f.name.lower()
        if "_pa_drr" in fname_lower:
            pa_drr = f
        elif "_lat_drr" in fname_lower or "_lateral_drr" in fname_lower:
            lat_drr = f
    
    for f in patient_dir.glob("*.nii.gz"):
        if "_reconstructed" not in f.name:
            gt_ct = f
            break
    
    return pa_drr, lat_drr, gt_ct


def process_patient(model, patient_dir, output_dir, device, compare_gt=False):
    """
    Process a single patient's DRR data.
    
    Returns:
        dict with keys: 'success', 'psnr', 'ssim' (metrics are None if no GT available)
    """
    patient_dir = Path(patient_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_id = patient_dir.name
    result = {'success': False, 'psnr': None, 'ssim': None}
    
    # Find files
    pa_drr, lat_drr, gt_ct_path = find_patient_files(patient_dir)
    
    if pa_drr is None or lat_drr is None:
        print(f"  ⚠ Skipping {patient_id}: Missing PA or Lateral DRR")
        return result
    
    print(f"\n{'='*50}")
    print(f"Patient: {patient_id}")
    print(f"  PA DRR: {pa_drr.name}")
    print(f"  Lateral DRR: {lat_drr.name}")
    
    # Load and preprocess DRRs
    # Note: Your DRRs are already flipped (*_pa_drr_flipped.png), so we check for that
    # If files end with _flipped, the flip was already applied externally
    already_flipped = "_flipped" in pa_drr.name.lower()
    
    pa_img = load_and_preprocess_drr(pa_drr, apply_camera_transform=not already_flipped, camera_type='PA')
    lat_img = load_and_preprocess_drr(lat_drr, apply_camera_transform=not already_flipped, camera_type='Lateral')
    
    # Convert to tensors (keep as H, W, 3 - model's get_input will permute)
    pa_tensor = torch.from_numpy(pa_img).unsqueeze(0).float().to(device)  # (1, H, W, 3)
    lat_tensor = torch.from_numpy(lat_img).unsqueeze(0).float().to(device)  # (1, H, W, 3)
    
    # Reconstruct 3D volume
    global INFERENCE_RESOLUTION
    num_slices = INFERENCE_RESOLUTION if 'INFERENCE_RESOLUTION' in globals() else 128
    print(f"  Reconstructing 3D volume ({num_slices} slices)...")
    volume = reconstruct_ct_volume(model, pa_tensor, lat_tensor, device, num_slices=num_slices)
    
    print(f"  ✓ Volume shape: {volume.shape}")
    print(f"  ✓ Value range: [{volume.min():.3f}, {volume.max():.3f}]")
    
    # Rescale to approximate HU values for better visualization
    # Training normalized to [0,1], so we rescale back
    volume_hu = volume * 2500  # CT_MIN_MAX was [0, 2500]
    
    # Save as NIfTI
    output_path = output_dir / f"{patient_id}_reconstructed.nii.gz"
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(volume_hu.astype(np.float32), affine)
    nib.save(nii_img, str(output_path))
    print(f"  ✓ Saved: {output_path.name}")
    
    # Save input DRRs visualization
    pa_vis = (pa_img[:, :, 0] * 255).astype(np.uint8)
    lat_vis = (lat_img[:, :, 0] * 255).astype(np.uint8)
    combined_drr = np.concatenate([pa_vis, lat_vis], axis=1)
    drr_path = output_dir / f"{patient_id}_input_drrs.png"
    imageio.imwrite(str(drr_path), combined_drr)
    
    # Save middle slice preview
    mid_slice = volume_hu[:, :, volume_hu.shape[2] // 2]
    mid_slice_vis = ((mid_slice - mid_slice.min()) / (mid_slice.max() - mid_slice.min() + 1e-8) * 255).astype(np.uint8)
    preview_path = output_dir / f"{patient_id}_middle_slice.png"
    imageio.imwrite(str(preview_path), mid_slice_vis)
    print(f"  ✓ Saved preview: {preview_path.name}")
    
    # Compare with ground truth if available
    if compare_gt and gt_ct_path and gt_ct_path.exists():
        try:
            gt_nii = nib.load(gt_ct_path)
            gt_ct = gt_nii.get_fdata()
            
            # Resize GT to match reconstructed volume if needed
            if gt_ct.shape != volume_hu.shape:
                gt_ct_resized = resize(gt_ct, volume_hu.shape, preserve_range=True, anti_aliasing=True)
            else:
                gt_ct_resized = gt_ct
            
            # Normalize both volumes to [0, 1] for metric computation
            recon_norm = (volume_hu - volume_hu.min()) / (volume_hu.max() - volume_hu.min() + 1e-8)
            gt_norm = (gt_ct_resized - gt_ct_resized.min()) / (gt_ct_resized.max() - gt_ct_resized.min() + 1e-8)
            
            # Compute 3D PSNR
            psnr_3d = psnr(gt_norm, recon_norm, data_range=1.0)
            
            # Compute 3D SSIM (compute per slice and average for memory efficiency)
            ssim_slices = []
            for s in range(volume_hu.shape[2]):
                ssim_val = ssim(gt_norm[:, :, s], recon_norm[:, :, s], data_range=1.0)
                ssim_slices.append(ssim_val)
            ssim_3d = np.mean(ssim_slices)
            
            result['psnr'] = psnr_3d
            result['ssim'] = ssim_3d
            
            print(f"  ✓ PSNR: {psnr_3d:.2f} dB")
            print(f"  ✓ SSIM: {ssim_3d:.4f}")
            
            # Get middle slices for comparison visualization
            gt_mid = gt_ct[:, :, gt_ct.shape[2] // 2]
            if gt_mid.shape != mid_slice.shape:
                gt_mid = resize(gt_mid, mid_slice.shape, preserve_range=True)
            
            # Normalize for visualization
            gt_vis = ((gt_mid - gt_mid.min()) / (gt_mid.max() - gt_mid.min() + 1e-8) * 255).astype(np.uint8)
            
            # Side by side comparison with metrics annotation
            comparison = np.concatenate([mid_slice_vis, gt_vis], axis=1)
            comp_path = output_dir / f"{patient_id}_comparison.png"
            imageio.imwrite(str(comp_path), comparison)
            print(f"  ✓ Saved comparison: {comp_path.name}")
            
        except Exception as e:
            print(f"  ⚠ Could not load GT for comparison: {e}")
    
    result['success'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description='PerX2CT Inference - Reconstruct CT from DRRs')
    parser.add_argument('--config_path', type=str, default='./configs/PerX2CT_custom.yaml',
                        help='Path to config file')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing patient folders with DRR images')
    parser.add_argument('--output_dir', type=str, default='./inference_outputs',
                        help='Directory to save reconstructed volumes')
    parser.add_argument('--resolution', type=int, default=128, choices=[128, 256],
                        help='Input/output resolution (128 or 256)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--compare_gt', action='store_true',
                        help='Compare with ground truth CT if available')
    parser.add_argument('--patient', type=str, default=None,
                        help='Process only a specific patient folder (optional)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PerX2CT Inference - DRR to CT Reconstruction")
    print("="*60)
    print(f"Config: {args.config_path}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print("="*60)
    
    # Store resolution globally for use in process_patient
    global INFERENCE_RESOLUTION
    INFERENCE_RESOLUTION = args.resolution
    
    # Load model
    model, config = load_model(args.config_path, args.ckpt_path, args.device)
    print("✓ Model loaded successfully")
    
    # Find patient directories
    data_dir = Path(args.data_dir)
    
    if args.patient:
        # Process single patient
        patient_dirs = [data_dir / args.patient]
        if not patient_dirs[0].exists():
            print(f"Error: Patient directory not found: {patient_dirs[0]}")
            sys.exit(1)
    else:
        # Process all patients
        patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"\nFound {len(patient_dirs)} patient(s) to process")
    
    # Process each patient
    results = []
    for patient_dir in patient_dirs:
        result = process_patient(model, patient_dir, args.output_dir, args.device, args.compare_gt)
        results.append((patient_dir.name, result))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    successful = sum(1 for _, r in results if r['success'])
    print(f"Successfully processed: {successful}/{len(results)} patients")
    
    # Compute average metrics if available
    psnr_values = [r['psnr'] for _, r in results if r['psnr'] is not None]
    ssim_values = [r['ssim'] for _, r in results if r['ssim'] is not None]
    
    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        std_psnr = np.std(psnr_values)
        avg_ssim = np.mean(ssim_values)
        std_ssim = np.std(ssim_values)
        
        print("\n" + "-"*40)
        print("METRICS (vs Ground Truth)")
        print("-"*40)
        print(f"  Patients evaluated: {len(psnr_values)}")
        print(f"  Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"  Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        print("-"*40)
        
        # Per-patient breakdown
        print("\nPer-patient metrics:")
        for patient_name, r in results:
            if r['psnr'] is not None:
                print(f"  {patient_name}: PSNR={r['psnr']:.2f} dB, SSIM={r['ssim']:.4f}")
        
        # Save metrics to CSV
        metrics_path = Path(args.output_dir) / "metrics.csv"
        with open(metrics_path, 'w') as f:
            f.write("patient,psnr_db,ssim\n")
            for patient_name, r in results:
                if r['psnr'] is not None:
                    f.write(f"{patient_name},{r['psnr']:.4f},{r['ssim']:.6f}\n")
            f.write(f"AVERAGE,{avg_psnr:.4f},{avg_ssim:.6f}\n")
            f.write(f"STD,{std_psnr:.4f},{std_ssim:.6f}\n")
        print(f"\n✓ Metrics saved to: {metrics_path}")
    
    if successful > 0:
        print(f"\nOutputs saved to: {args.output_dir}")
        print("Files per patient:")
        print("  - {patient}_reconstructed.nii.gz  (3D CT volume)")
        print("  - {patient}_input_drrs.png        (PA + Lateral DRRs)")
        print("  - {patient}_middle_slice.png      (Preview)")
        if args.compare_gt:
            print("  - {patient}_comparison.png        (Recon vs GT)")
            print("  - metrics.csv                     (All patient metrics)")


if __name__ == "__main__":
    main()
