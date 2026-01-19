#!/usr/bin/env python3
"""
Full Resolution CT Volume Inference Script
Generates 512^3 CT volumes from bidirectional DRR images (PA + Lateral)

This script:
1. Loads trained PerX2CT model
2. Generates 128^3 CT volume from DRRs
3. Upscales to 512^3 using high-quality interpolation
4. Saves as NIfTI with proper spacing

Usage:
    python inference_full_volume.py \
        --ckpt_path ./logs/PerX2CT/custom_3500/.../best_psnr.ckpt \
        --pa_drr /path/to/patient_pa_drr.png \
        --lat_drr /path/to/patient_lat_drr.png \
        --output ./output_patient.nii.gz \
        --output_size 512
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
import math
from scipy.ndimage import zoom as scipy_zoom
from skimage.transform import resize as skimage_resize


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


def load_and_preprocess_drr(drr_path, target_size=128, apply_camera_transform=True, camera_type='PA'):
    """
    Load and preprocess a DRR image to match training data format.
    """
    img = Image.open(drr_path).convert('L')
    
    # Resize if needed
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.BILINEAR)
    
    img_array = np.array(img, dtype=np.float32)
    
    # Apply camera-specific transformations
    if apply_camera_transform:
        if camera_type == 'PA':
            img_array = np.fliplr(img_array)
        elif camera_type == 'Lateral':
            img_array = np.transpose(img_array, (1, 0))
            img_array = np.flipud(img_array)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Create 3-channel image
    img_array = np.expand_dims(img_array, -1)
    img_array = np.concatenate([img_array, img_array, img_array], axis=-1)
    
    return img_array


def reconstruct_base_volume(model, pa_tensor, lat_tensor, device, num_slices=128):
    """
    Reconstruct base resolution 3D CT volume (128^3).
    """
    pa_cam = torch.tensor([[0.0, 0.0]], device=device)
    lateral_cam = torch.tensor([[math.pi / 2, math.pi / 2]], device=device)
    
    volume_slices = []
    
    for slice_idx in tqdm(range(num_slices), desc="Reconstructing base volume", leave=True):
        dummy_ct = torch.zeros_like(pa_tensor)
        
        batch = {
            'ctslice': dummy_ct,
            'PA': pa_tensor,
            'Lateral': lat_tensor,
            'PA_cam': pa_cam,
            'Lateral_cam': lateral_cam,
            'file_path_': [f"inference/axial_{slice_idx:03d}.h5"],
            'image_key': 'ctslice',
        }
        
        batch = model.get_input(batch)
        batch['image_key'] = 'ctslice'
        
        with torch.no_grad():
            output_dict, _ = model(batch)
        
        recon = output_dict['outputs']
        recon_np = recon[0, 0].cpu().numpy()
        
        volume_slices.append(recon_np)
    
    volume = np.stack(volume_slices, axis=2)
    return volume


def upscale_volume(volume, target_size=512, method='spline'):
    """
    Upscale volume from base resolution to target resolution.
    
    Args:
        volume: Input volume (H, W, D)
        target_size: Target size for each dimension
        method: 'spline' (scipy) or 'linear' (skimage)
    
    Returns:
        Upscaled volume of shape (target_size, target_size, target_size)
    """
    current_size = volume.shape[0]
    scale_factor = target_size / current_size
    
    print(f"\nUpscaling volume: {volume.shape} -> ({target_size}, {target_size}, {target_size})")
    print(f"Scale factor: {scale_factor:.2f}x")
    
    if method == 'spline':
        # Use scipy's zoom with cubic spline interpolation
        upscaled = scipy_zoom(volume, scale_factor, order=3, mode='nearest')
    else:
        # Use skimage's resize with anti-aliasing
        upscaled = skimage_resize(
            volume, 
            (target_size, target_size, target_size),
            order=3,  # Cubic interpolation
            preserve_range=True,
            anti_aliasing=True
        )
    
    return upscaled.astype(np.float32)


def enhance_volume(volume, sharpen=True):
    """
    Optional post-processing to enhance the upscaled volume.
    """
    if sharpen:
        from scipy.ndimage import gaussian_filter, laplace
        
        # Unsharp masking for edge enhancement
        blurred = gaussian_filter(volume, sigma=1.0)
        mask = volume - blurred
        enhanced = volume + 0.5 * mask  # Subtle sharpening
        
        return np.clip(enhanced, volume.min(), volume.max())
    
    return volume


def main():
    parser = argparse.ArgumentParser(description='Generate Full Resolution CT Volume from DRRs')
    parser.add_argument('--config_path', type=str, default='./configs/PerX2CT_custom.yaml',
                        help='Path to config file')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--pa_drr', type=str, required=True,
                        help='Path to PA (frontal) DRR image')
    parser.add_argument('--lat_drr', type=str, required=True,
                        help='Path to Lateral DRR image')
    parser.add_argument('--output', type=str, default='./reconstructed_ct.nii.gz',
                        help='Output NIfTI file path')
    parser.add_argument('--output_size', type=int, default=512,
                        help='Output volume size (default: 512 for 512^3)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--no_transform', action='store_true',
                        help='Skip camera transforms (use if DRRs are already transformed)')
    parser.add_argument('--sharpen', action='store_true',
                        help='Apply sharpening to upscaled volume')
    parser.add_argument('--spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='Voxel spacing in mm (x, y, z)')
    parser.add_argument('--upscale_method', type=str, default='spline', choices=['spline', 'linear'],
                        help='Interpolation method for upscaling')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PerX2CT Full Resolution CT Reconstruction")
    print("="*70)
    print(f"PA DRR:      {args.pa_drr}")
    print(f"Lateral DRR: {args.lat_drr}")
    print(f"Output:      {args.output}")
    print(f"Output size: {args.output_size}^3")
    print("="*70)
    
    # Check inputs exist
    if not os.path.exists(args.pa_drr):
        print(f"Error: PA DRR not found: {args.pa_drr}")
        sys.exit(1)
    if not os.path.exists(args.lat_drr):
        print(f"Error: Lateral DRR not found: {args.lat_drr}")
        sys.exit(1)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, config = load_model(args.config_path, args.ckpt_path, args.device)
    print("✓ Model loaded successfully")
    
    # Load and preprocess DRRs
    print("\n[2/4] Loading DRR images...")
    already_flipped = "_flipped" in args.pa_drr.lower() or args.no_transform
    
    pa_img = load_and_preprocess_drr(
        args.pa_drr, 
        apply_camera_transform=not already_flipped, 
        camera_type='PA'
    )
    lat_img = load_and_preprocess_drr(
        args.lat_drr, 
        apply_camera_transform=not already_flipped, 
        camera_type='Lateral'
    )
    
    pa_tensor = torch.from_numpy(pa_img).unsqueeze(0).float().to(args.device)
    lat_tensor = torch.from_numpy(lat_img).unsqueeze(0).float().to(args.device)
    print(f"✓ PA DRR loaded: {pa_img.shape[:2]}")
    print(f"✓ Lateral DRR loaded: {lat_img.shape[:2]}")
    
    # Reconstruct base volume (128^3)
    print("\n[3/4] Reconstructing base volume (128^3)...")
    base_volume = reconstruct_base_volume(model, pa_tensor, lat_tensor, args.device, num_slices=128)
    print(f"✓ Base volume: {base_volume.shape}")
    print(f"  Value range: [{base_volume.min():.4f}, {base_volume.max():.4f}]")
    
    # Rescale to HU values
    base_volume_hu = base_volume * 2500  # CT_MIN_MAX was [0, 2500]
    
    # Upscale to target resolution
    print(f"\n[4/4] Upscaling to {args.output_size}^3...")
    if args.output_size == 128:
        full_volume = base_volume_hu
        print("✓ No upscaling needed (output_size = 128)")
    else:
        full_volume = upscale_volume(base_volume_hu, args.output_size, method=args.upscale_method)
        print(f"✓ Upscaled volume: {full_volume.shape}")
    
    # Optional enhancement
    if args.sharpen and args.output_size > 128:
        print("  Applying edge enhancement...")
        full_volume = enhance_volume(full_volume, sharpen=True)
        print("✓ Enhancement applied")
    
    print(f"\n  Final value range: [{full_volume.min():.1f}, {full_volume.max():.1f}] HU")
    
    # Create NIfTI with proper affine
    print("\nSaving volume...")
    spacing = args.spacing
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    
    nii_img = nib.Nifti1Image(full_volume.astype(np.float32), affine)
    nii_img.header.set_xyzt_units('mm')
    nii_img.header['descrip'] = f'PerX2CT reconstructed CT volume {args.output_size}^3'
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    nib.save(nii_img, str(output_path))
    
    # Print summary
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print("\n" + "="*70)
    print("RECONSTRUCTION COMPLETE")
    print("="*70)
    print(f"Output file:    {args.output}")
    print(f"Volume shape:   {full_volume.shape}")
    print(f"Voxel spacing:  {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm")
    print(f"Value range:    [{full_volume.min():.1f}, {full_volume.max():.1f}] HU")
    print(f"File size:      {file_size_mb:.1f} MB")
    print("="*70)
    print("\nTo view the result:")
    print(f"  - 3D Slicer: File -> Add Data -> {args.output}")
    print(f"  - ITK-SNAP: File -> Open Main Image")
    print(f"  - Python:   nib.load('{args.output}').get_fdata()")


if __name__ == "__main__":
    main()
