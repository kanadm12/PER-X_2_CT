"""
Simple inference script for DRR patient data
This script loads a trained PerX2CT model and generates CT volumes from DRR images
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
    """Load and preprocess a DRR image"""
    img = Image.open(drr_path).convert('L')  # Convert to grayscale
    
    # Resize if needed
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1]
    if img_array.max() > 0:
        img_array = img_array / 255.0
    
    return img_array


def run_inference_on_patient(model, patient_dir, output_dir, device='cuda'):
    """
    Run inference on a single patient's DRR data
    
    Args:
        model: Trained PerX2CT model
        patient_dir: Directory containing patient DRR files (PA and Lateral)
        output_dir: Directory to save output
        device: Device to run inference on
    """
    patient_dir = Path(patient_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_id = patient_dir.name
    
    # Find DRR files
    pa_drr = None
    lat_drr = None
    
    for file in patient_dir.glob("*.png"):
        if "_pa_drr" in file.name or "_PA_drr" in file.name:
            pa_drr = file
        elif "_lat_drr" in file.name or "_lateral_drr" in file.name or "_Lateral_drr" in file.name:
            lat_drr = file
    
    if pa_drr is None or lat_drr is None:
        print(f"Warning: Could not find both PA and Lateral DRRs for {patient_id}")
        print(f"  PA DRR: {pa_drr}")
        print(f"  Lateral DRR: {lat_drr}")
        return None
    
    print(f"\nProcessing patient: {patient_id}")
    print(f"  PA DRR: {pa_drr.name}")
    print(f"  Lateral DRR: {lat_drr.name}")
    
    # Load and preprocess DRRs
    pa_img = preprocess_drr(pa_drr)
    lat_img = preprocess_drr(lat_drr)
    
    # Prepare input
    # Stack to 3 channels (model expects 3-channel input)
    pa_tensor = torch.from_numpy(pa_img).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    lat_tensor = torch.from_numpy(lat_img).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    
    # Run inference
    with torch.no_grad():
        try:
            # This is a simplified version - you may need to adjust based on actual model input format
            # The model expects specific input format with PA and Lateral views
            batch = {
                'PA': pa_tensor,
                'Lateral': lat_tensor,
            }
            
            # Generate CT reconstruction
            output = model(batch)
            
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
            
            # Save as NIfTI
            output_path = output_dir / f"{patient_id}_reconstructed_ct.nii.gz"
            nii_img = nib.Nifti1Image(recon_np, np.eye(4))
            nib.save(nii_img, str(output_path))
            
            print(f"  ✓ Saved reconstruction to: {output_path}")
            
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
    
    args = parser.parse_args()
    
    # Load model
    print("="*60)
    print("PerX2CT DRR Inference")
    print("="*60)
    model, config = load_model(args.config_path, args.ckpt_path, args.device)
    
    # Find all patient directories
    data_dir = Path(args.data_dir)
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    print(f"\nFound {len(patient_dirs)} patient directories")
    print(f"Output directory: {args.output_dir}\n")
    
    # Process each patient
    results = []
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        result = run_inference_on_patient(
            model, patient_dir, args.output_dir, args.device
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
