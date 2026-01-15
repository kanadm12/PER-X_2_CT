"""
Simple inference script for DRR patient data
This script loads a trained PerX2CT model and generates CT volumes from DRR images

Data format expected:
- Each patient folder contains:
  - *_pa_drr.png: PA (posterior-anterior) view DRR
  - *_lat_drr.png: Lateral view DRR  
  - *.nii.gz: Ground truth CT volume (optional, for comparison)
"""

import os
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


def load_model(config_path, ckpt_path, device='cuda'):
    """Load the trained model"""
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


def preprocess_drr(drr_path, target_size=128):
    """Load and preprocess a DRR image to match training data format"""
    img = Image.open(drr_path).convert('L')  # Convert to grayscale
    
    # Resize if needed
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to numpy array (H, W)
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1] range (matching training data preprocessing)
    # Training uses: Normalization(XRAY_MIN_MAX[0], XRAY_MIN_MAX[1]) where XRAY_MIN_MAX = [0, 255]
    img_array = img_array / 255.0
    
    # Create 3-channel image by replicating grayscale (H, W, 3) format
    # Training data does: np.concatenate((image, image, image), axis=-1)
    img_array = np.expand_dims(img_array, -1)  # (H, W, 1)
    img_array = np.concatenate([img_array, img_array, img_array], axis=-1)  # (H, W, 3)
    
    return img_array


def run_inference_on_patient(model, patient_dir, output_dir, device='cuda', compare_gt=False, num_slices=128):
    """
    Run inference on a single patient's DRR data
    
    Args:
        model: Trained PerX2CT model
        patient_dir: Directory containing patient DRR files (PA and Lateral) and optional GT CT
        output_dir: Directory to save output
        device: Device to run inference on
        compare_gt: If True, load ground truth CT and save comparison images
        num_slices: Number of CT slices to reconstruct (default: 128 for full volume)
    """
    patient_dir = Path(patient_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_id = patient_dir.name
    
    # Find DRR files (these should be the vertically flipped versions)
    pa_drr = None
    lat_drr = None
    gt_ct_path = None
    
    for file in patient_dir.glob("*.png"):
        if "_pa_drr" in file.name.lower():
            pa_drr = file
        elif "_lat_drr" in file.name.lower() or "_lateral_drr" in file.name.lower():
            lat_drr = file
    
    # Find ground truth CT
    for file in patient_dir.glob("*.nii.gz"):
        if not ("_reconstructed" in file.name or "_flipped" in file.name):
            gt_ct_path = file
            break
    
    if pa_drr is None or lat_drr is None:
        print(f"Warning: Could not find both PA and Lateral DRRs for {patient_id}")
        print(f"  PA DRR: {pa_drr}")
        print(f"  Lateral DRR: {lat_drr}")
        return None
    
    print(f"\nProcessing patient: {patient_id}")
    print(f"  PA DRR: {pa_drr.name}")
    print(f"  Lateral DRR: {lat_drr.name}")
    if gt_ct_path:
        print(f"  Ground Truth CT: {gt_ct_path.name}")
    
    # Load and preprocess DRRs (already flipped by flip_drr_images.py script)
    pa_img = preprocess_drr(pa_drr)  # Returns (H, W, 3) numpy array
    lat_img = preprocess_drr(lat_drr)  # Returns (H, W, 3) numpy array
    
    # Load ground truth CT if available and requested
    gt_ct = None
    if compare_gt and gt_ct_path and gt_ct_path.exists():
        try:
            gt_nii = nib.load(gt_ct_path)
            gt_ct = gt_nii.get_fdata()
            print(f"  Loaded GT CT shape: {gt_ct.shape}")
        except Exception as e:
            print(f"  Warning: Could not load GT CT: {e}")
    
    # Convert to torch tensors in (B, H, W, C) format (NOT (B, C, H, W))
    # The model's get_input() will do permute(0, 3, 1, 2) to convert to (B, C, H, W)
    # So we keep data in (H, W, C) format and just add batch dimension
    pa_tensor = torch.from_numpy(pa_img).unsqueeze(0).to(device).float()  # (1, H, W, 3)
    lat_tensor = torch.from_numpy(lat_img).unsqueeze(0).to(device).float()  # (1, H, W, 3)
    
    print(f"  Input tensor shapes - PA: {pa_tensor.shape}, Lateral: {lat_tensor.shape}")
    
    # Run inference
    with torch.no_grad():
        try:
            # Prepare camera poses (same for all slices)
            pa_cam = torch.tensor([[0.0, 0.0]], device=device)  # [batch_size, 2]
            lateral_cam = torch.tensor([[math.pi / 2, math.pi / 2]], device=device)  # [batch_size, 2]
            
            print(f"  Reconstructing {num_slices} CT slices...")
            reconstructed_volume = []
            
            # Iterate through slice positions
            for slice_idx in range(num_slices):
                # Create batch for this slice
                dummy_ct = torch.zeros_like(pa_tensor)  # Placeholder CT slice
                
                # File path with slice number (format: axis_slicenum.h5)
                slice_filepath = f"dummy_path/coronal_{slice_idx:03d}.h5"
                
                batch = {
                    'image_key': 'ctslice',
                    'ctslice': dummy_ct,
                    'PA': pa_tensor,
                    'Lateral': lat_tensor,
                    'PA_cam': pa_cam,
                    'Lateral_cam': lateral_cam,
                    'file_path_': [slice_filepath],
                }
                
                # Run inference for this slice
                output = model.log_images(batch, split='val', p0=None, zoom_size=None)
                
                # Extract reconstruction
                if isinstance(output, dict) and 'reconstructions' in output:
                    recon = output['reconstructions']
                    recon_np = recon.cpu().numpy()
                    
                    # Extract 2D slice: (1, 3, H, W) -> (H, W)
                    if len(recon_np.shape) == 4:
                        recon_slice = recon_np[0, 0, :, :]
                    elif len(recon_np.shape) == 3:
                        recon_slice = recon_np[0, :, :]
                    else:
                        recon_slice = recon_np
                    
                    reconstructed_volume.append(recon_slice)
                
                # Progress indicator
                if (slice_idx + 1) % 10 == 0 or slice_idx == 0:
                    print(f"    Progress: {slice_idx + 1}/{num_slices} slices")
            
            # Stack slices into 3D volume
            volume_3d = np.stack(reconstructed_volume, axis=2)  # (H, W, D)
            print(f"  ✓ Reconstructed 3D CT volume shape: {volume_3d.shape}")
            
            # Save reconstructed volume as NIfTI
            output_path = output_dir / f"{patient_id}_reconstructed_volume.nii.gz"
            nii_img = nib.Nifti1Image(volume_3d, np.eye(4))
            nib.save(nii_img, str(output_path))
            print(f"  ✓ Saved 3D volume to: {output_path.name}")
            
            # Also save middle slice separately
            mid_slice_idx = num_slices // 2
            mid_slice = reconstructed_volume[mid_slice_idx]
            mid_slice_path = output_dir / f"{patient_id}_middle_slice.nii.gz"
            nii_mid = nib.Nifti1Image(mid_slice, np.eye(4))
            nib.save(nii_mid, str(mid_slice_path))
            print(f"  ✓ Saved middle slice to: {mid_slice_path.name}")
            
            # Save comparison with GT if available
            if gt_ct is not None and compare_gt:
                # Use middle slice for comparison
                recon_slice = reconstructed_volume[mid_slice_idx]
                
                # Get corresponding GT slice (GT is typically H x W x D)
                gt_mid = gt_ct.shape[2] // 2
                gt_slice = gt_ct[:, :, gt_mid]
                
                # Normalize for visualization
                recon_vis = ((recon_slice - recon_slice.min()) / (recon_slice.max() - recon_slice.min() + 1e-8) * 255).astype(np.uint8)
                gt_vis = ((gt_slice - gt_slice.min()) / (gt_slice.max() - gt_slice.min() + 1e-8) * 255).astype(np.uint8)
                
                # Resize GT to match reconstruction size if needed
                if gt_vis.shape != recon_vis.shape:
                    from skimage.transform import resize
                    gt_vis = resize(gt_vis, recon_vis.shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)
                
                # Concatenate side by side (reconstruction | ground truth)
                comparison = np.concatenate([recon_vis, gt_vis], axis=1)
                
                comparison_path = output_dir / f"{patient_id}_comparison_middle_slice.png"
                imageio.imwrite(str(comparison_path), comparison)
                print(f"  ✓ Saved comparison to: {comparison_path.name}")
            
            # Save input DRR visualizations
            # pa_img and lat_img are (H, W, 3) numpy arrays in [0, 1] range
            drr_vis_pa = (pa_img[:, :, 0] * 255).astype(np.uint8)  # Take first channel
            drr_vis_lat = (lat_img[:, :, 0] * 255).astype(np.uint8)
            drr_combined = np.concatenate([drr_vis_pa, drr_vis_lat], axis=1)
            drr_path = output_dir / f"{patient_id}_input_drrs.png"
            imageio.imwrite(str(drr_path), drr_combined)
            print(f"  ✓ Saved input DRRs to: {drr_path.name}")
            
            return recon_np
            
        except Exception as e:
            print(f"  ✗ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    parser = argparse.ArgumentParser(description='Run PerX2CT inference on DRR patient data')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to model config file')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing patient folders with DRR images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save reconstructed CT volumes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--compare_gt', action='store_true',
                        help='If set, compare with ground truth CT volumes')
    parser.add_argument('--num_slices', type=int, default=128,
                        help='Number of CT slices to reconstruct for 3D volume (default: 128)')
    
    args = parser.parse_args()
    
    # Load model
    print("="*60)
    print("PerX2CT DRR to CT Reconstruction")
    print("="*60)
    print(f"Note: Using vertically flipped DRRs from {args.data_dir}")
    print("="*60)
    model, config = load_model(args.config_path, args.ckpt_path, args.device)
    
    # Find all patient directories
    data_dir = Path(args.data_dir)
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    print(f"\nFound {len(patient_dirs)} patient directories")
    print(f"Output directory: {args.output_dir}")
    print(f"Reconstructing {args.num_slices} slices per volume")
    if args.compare_gt:
        print(f"Ground truth comparison: ENABLED")
    print()
    
    # Process each patient
    results = []
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        result = run_inference_on_patient(
            model, patient_dir, args.output_dir, args.device, args.compare_gt, args.num_slices
        )
        results.append((patient_dir.name, result is not None))
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    successful = sum(1 for _, success in results if success)
    print(f"Successfully processed: {successful}/{len(results)} patients")
    
    failed = [(name, success) for name, success in results if not success]
    if failed:
        print(f"\nFailed patients:")
        for name, _ in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
