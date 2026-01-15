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


def run_inference_on_patient(model, patient_dir, output_dir, device='cuda', compare_gt=False):
    """
    Run inference on a single patient's DRR data
    
    Args:
        model: Trained PerX2CT model
        patient_dir: Directory containing patient DRR files (PA and Lateral) and optional GT CT
        output_dir: Directory to save output
        device: Device to run inference on
        compare_gt: If True, load ground truth CT and save comparison images
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
            # Prepare batch in the format expected by the model
            # The model expects 'image_key' to specify which key contains the CT data
            # Since we don't have GT CT (we're reconstructing it), we'll use a dummy placeholder
            dummy_ct = torch.zeros_like(pa_tensor)  # Placeholder CT slice
            
            # The model expects file_path_ to be in format: .../axis_slicenum.h5
            # e.g., "coronal_064.h5" where axis is the reconstruction axis and 064 is slice number
            dummy_filepath = f"dummy_path/coronal_064.h5"
            
            # Camera poses for PA and Lateral views (from x2ct_nerf/data/base.py)
            # PA: pitch=0, yaw=0 (frontal view)
            # Lateral: pitch=90°, yaw=90° (side view)
            pa_cam = torch.tensor([[0.0, 0.0]], device=device)  # [batch_size, 2]
            lateral_cam = torch.tensor([[math.pi / 2, math.pi / 2]], device=device)  # [batch_size, 2]
            
            batch = {
                'image_key': 'ctslice',  # Key name for CT data
                'ctslice': dummy_ct,     # Dummy CT (not used during inference)
                'PA': pa_tensor,         # PA view X-ray
                'Lateral': lat_tensor,   # Lateral view X-ray
                'PA_cam': pa_cam,        # PA camera pose (pitch, yaw)
                'Lateral_cam': lateral_cam,  # Lateral camera pose (pitch, yaw)
                'file_path_': [dummy_filepath],  # Dummy file path in expected format
            }
            
            print(f"  Batch PA shape: {batch['PA'].shape}, Lateral shape: {batch['Lateral'].shape}")
            print(f"  Batch PA_cam shape: {batch['PA_cam'].shape}, Lateral_cam shape: {batch['Lateral_cam'].shape}")
            print(f"  PA tensor min/max: {batch['PA'].min():.4f}/{batch['PA'].max():.4f}")
            print(f"  Lateral tensor min/max: {batch['Lateral'].min():.4f}/{batch['Lateral'].max():.4f}")
            
            # Use log_images method for inference (this is what the test script uses)
            output = model.log_images(batch, split='val', p0=None, zoom_size=None)
            
            # Save output
            # Note: This is a placeholder - actual output format depends on model implementation
            if isinstance(output, dict):
                if 'reconstructions' in output:
                    recon = output['reconstructions']
                elif 'rec' in output:
                    recon = output['rec']
                else:
                    recon = list(output.values())[0]
            else:
                recon = output
            
            # Convert to numpy and save
            recon_np = recon.cpu().numpy()
            
            # Save reconstructed CT as NIfTI
            output_path = output_dir / f"{patient_id}_reconstructed_ct.nii.gz"
            nii_img = nib.Nifti1Image(recon_np, np.eye(4))
            nib.save(nii_img, str(output_path))
            
            print(f"  ✓ Saved reconstruction to: {output_path.name}")
            print(f"    Reconstructed CT shape: {recon_np.shape}")
            
            # Save comparison slices if GT is available
            if gt_ct is not None:
                # Save middle slice comparison
                mid_slice = recon_np.shape[0] // 2 if len(recon_np.shape) > 2 else 0
                
                if len(recon_np.shape) >= 3:
                    recon_slice = recon_np[mid_slice, :, :]
                else:
                    recon_slice = recon_np[0, :, :] if len(recon_np.shape) == 3 else recon_np
                
                # Get corresponding GT slice
                if len(gt_ct.shape) >= 3:
                    gt_mid = gt_ct.shape[0] // 2
                    gt_slice = gt_ct[gt_mid, :, :]
                else:
                    gt_slice = gt_ct
                
                # Normalize for visualization
                recon_vis = ((recon_slice - recon_slice.min()) / (recon_slice.max() - recon_slice.min() + 1e-8) * 255).astype(np.uint8)
                gt_vis = ((gt_slice - gt_slice.min()) / (gt_slice.max() - gt_slice.min() + 1e-8) * 255).astype(np.uint8)
                
                # Resize if needed
                if gt_vis.shape != recon_vis.shape:
                    from skimage.transform import resize
                    gt_vis = resize(gt_vis, recon_vis.shape, preserve_range=True).astype(np.uint8)
                
                # Concatenate side by side
                comparison = np.concatenate([recon_vis, gt_vis], axis=1)
                
                comparison_path = output_dir / f"{patient_id}_comparison_slice_{mid_slice}.png"
                imageio.imwrite(str(comparison_path), comparison)
                print(f"  ✓ Saved comparison to: {comparison_path.name}")
            
            # Save input DRR visualizations
            drr_vis_pa = (pa_img * 255).astype(np.uint8)
            drr_vis_lat = (lat_img * 255).astype(np.uint8)
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
    if args.compare_gt:
        print(f"Ground truth comparison: ENABLED")
    print()
    
    # Process each patient
    results = []
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        result = run_inference_on_patient(
            model, patient_dir, args.output_dir, args.device, args.compare_gt
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
