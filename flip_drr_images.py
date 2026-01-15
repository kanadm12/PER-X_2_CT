"""
Script to vertically flip all DRR images in patient folders.
This script will process all PNG files (DRR images) found in subdirectories
of the drr_patient_data folder.
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def flip_drr_images(root_dir, backup=True, inplace=True):
    """
    Vertically flip all DRR PNG images in patient folders.
    
    Args:
        root_dir (str): Root directory containing patient folders (e.g., /workspace/drr_patient_data/)
        backup (bool): If True, create a backup of original images before flipping
        inplace (bool): If True, replace original images. If False, save to *_flipped.png
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Error: Directory {root_dir} does not exist!")
        return
    
    # Find all PNG files in subdirectories
    png_files = list(root_path.rglob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in {root_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files to process")
    
    processed = 0
    errors = 0
    
    for png_file in tqdm(png_files, desc="Flipping DRR images"):
        try:
            # Read the image
            img = Image.open(png_file)
            
            # Create backup if requested
            if backup and inplace:
                backup_path = png_file.with_suffix('.png.backup')
                if not backup_path.exists():
                    img.save(backup_path)
            
            # Vertically flip the image (flip top-to-bottom)
            flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Save the flipped image
            if inplace:
                output_path = png_file
            else:
                # Save with _flipped suffix
                output_path = png_file.with_name(png_file.stem + '_flipped.png')
            
            flipped_img.save(output_path)
            processed += 1
            
            # Close images to free memory
            img.close()
            flipped_img.close()
            
        except Exception as e:
            print(f"\nError processing {png_file}: {e}")
            errors += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully flipped: {processed} images")
    if errors > 0:
        print(f"Errors encountered: {errors} images")
    if backup and inplace:
        print(f"Backup files saved with .backup extension")
    print(f"{'='*60}")


def main():
    # Configuration
    ROOT_DIR = "/workspace/drr_patient_data"  # Change this path if needed
    
    print("="*60)
    print("DRR Image Vertical Flip Script")
    print("="*60)
    print(f"Root directory: {ROOT_DIR}")
    print()
    
    # You can modify these settings:
    # - backup=True: Creates .backup files before flipping
    # - inplace=True: Replaces original files (if False, creates *_flipped.png)
    
    flip_drr_images(
        root_dir=ROOT_DIR,
        backup=True,      # Create backups before flipping
        inplace=True      # Replace original files
    )


if __name__ == "__main__":
    main()
