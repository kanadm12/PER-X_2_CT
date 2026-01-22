"""
Custom data loader for 256x256 PerX2CT training on custom DRR patient data.

This loader handles the custom directory structure:
    CT slices:  /workspace/processed_ct256_CTSlice/patient_XXX/patient_XXX_NNN.h5
    X-rays:     /workspace/drr_patient_data_256/patient_XXX/patient_XXX_pa_drr_flipped.png
                /workspace/drr_patient_data_256/patient_XXX/patient_XXX_lat_drr_flipped.png
"""

import os
import re
import numpy as np
from torch.utils.data import Dataset
import torch
import math
import h5py
import imageio
from x2ct_nerf.preprocessing.X2CT_transform_3d import List_Compose, Limit_Min_Max_Threshold, Normalization, ToTensor
from pathlib import Path


class Custom256MultiInputDataset(Dataset):
    """
    Custom dataset for 256x256 DRR patient data.
    
    Unlike the original LIDC loader, this handles the simplified structure:
    - CT slices: /path/to/ct/patient_id/patient_id_000.h5
    - X-rays: /path/to/xray/patient_id/patient_id_pa_drr_flipped.png
    
    The mapping from CT path to X-ray path uses the CT directory's parent
    sibling structure instead of regex-based path manipulation.
    """
    
    def __init__(self, paths, opt: dict, size=None, labels=None, num_ctslice_per_item=1):
        assert num_ctslice_per_item in [1, 3]
        
        self.opt = opt
        self.ct_size = opt['ct_size']  # number of CT slices per patient
        self.xray_size = opt['xray_size']  # xray resolution (256)
        self.input_types = opt['input_type']
        
        # Custom paths for your directory structure
        self.xray_base_dir = opt.get('xray_base_dir', '/workspace/drr_patient_data_256')
        
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        self.num_ctslice_per_item = num_ctslice_per_item
        
        self.CT_MIN_MAX = opt["CT_MIN_MAX"]
        self.XRAY_MIN_MAX = opt["XRAY_MIN_MAX"]
        
        self.set_preprocessing()
        self.mapping_camera_type2pose = {
            "PA": torch.tensor([0, 0]),
            "Lateral": torch.tensor([math.pi / 2, math.pi / 2]),
        }
        
        # Build X-ray cache for efficient lookup
        self._build_xray_cache()
    
    def _build_xray_cache(self):
        """Pre-cache X-ray paths for all patients"""
        self.xray_cache = {}
        xray_dir = Path(self.xray_base_dir)
        
        if xray_dir.exists():
            for patient_dir in xray_dir.iterdir():
                if patient_dir.is_dir():
                    patient_id = patient_dir.name
                    
                    # Find PA and Lateral X-rays
                    pa_files = list(patient_dir.glob("*_pa_drr*.png"))
                    lat_files = list(patient_dir.glob("*_lat_drr*.png"))
                    
                    if pa_files and lat_files:
                        self.xray_cache[patient_id] = {
                            'PA': str(pa_files[0]),
                            'Lateral': str(lat_files[0])
                        }
        
        print(f"[Custom256] Built X-ray cache for {len(self.xray_cache)} patients")
    
    def __len__(self):
        return self._length
    
    def set_preprocessing(self):
        dict_augment_list = {}
        for input_type in self.input_types:
            i_type = 'ct' if input_type in ['ct', 'ctslice'] else 'xray'
            dict_augment_list[input_type] = self.opt[f"{i_type}_augment_list"]
        
        self.dict_preprocessing = {}
        for input_type in self.input_types:
            augment_list = []
            if input_type in ['ct', 'ctslice']:
                if 'min_max_th' in dict_augment_list[input_type]:
                    augment_list.append((Limit_Min_Max_Threshold(self.CT_MIN_MAX[0], self.CT_MIN_MAX[1]),))
                if 'normalization' in dict_augment_list[input_type]:
                    augment_list.append((Normalization(self.CT_MIN_MAX[0], self.CT_MIN_MAX[1]),))
            elif input_type in ['PA', 'Lateral']:
                if 'normalization' in dict_augment_list[input_type]:
                    augment_list.append((Normalization(self.XRAY_MIN_MAX[0], self.XRAY_MIN_MAX[1]),))
            augment_list.append((ToTensor(),))
            self.dict_preprocessing[input_type] = List_Compose(augment_list)
    
    def get_image(self, image_path, data_type='ct'):
        """Load image from file (H5 or PNG)"""
        ext = image_path.split(".")[-1]
        assert ext in ['png', 'h5'], f"Unsupported extension: {ext}"
        
        if ext == 'png':
            image = imageio.imread(image_path)
            image = np.asarray(image)
        elif ext == "h5":
            with h5py.File(image_path, 'r') as f:
                # Try 'ct' key first, fallback to 'ctslice'
                if data_type in f:
                    image = np.asarray(f[data_type])
                elif 'ct' in f:
                    image = np.asarray(f['ct'])
                elif 'ctslice' in f:
                    image = np.asarray(f['ctslice'])
                else:
                    raise KeyError(f"No valid key found in {image_path}. Keys: {list(f.keys())}")
        
        return image
    
    def apply_preprocessing_xray_according2cam(self, xray_img, src_camtype):
        """Apply camera-specific preprocessing to X-ray images"""
        assert src_camtype in ["PA", "Lateral"]
        if src_camtype == "PA":
            xray_img = np.fliplr(xray_img)
        elif src_camtype == "Lateral":
            xray_img = np.transpose(xray_img, (1, 0))
            xray_img = np.flipud(xray_img)
        src_campose = self.mapping_camera_type2pose[src_camtype]
        return xray_img, src_campose
    
    def get_ctslice(self, image_path):
        """Load CT slice(s) from H5 file"""
        if self.num_ctslice_per_item == 1:
            image = self.get_image(image_path)
            image = np.expand_dims(image, -1)
            image = np.concatenate((image, image, image), axis=-1)
        else:  # num_ctslice_per_item == 3
            # Extract slice index from filename: patient_XXX_NNN.h5 -> NNN
            filename = os.path.basename(image_path)
            match = re.search(r'_(\d{3})\.h5$', filename)
            if match:
                curr_slice_id = match.group(1)
            else:
                # Fallback pattern
                curr_slice_id = image_path.split("_")[-1].split(".")[0]
            
            start_i = int(curr_slice_id) - (self.num_ctslice_per_item - 1) // 2
            image = None
            
            for i in range(self.num_ctslice_per_item):
                slice_i = start_i + i
                if slice_i < 0:
                    slice_i_str = f"{0:03d}"
                elif slice_i > (self.ct_size - 1):
                    slice_i_str = f"{(self.ct_size - 1):03d}"
                else:
                    slice_i_str = f"{slice_i:03d}"
                
                img_path = image_path.replace(f"_{curr_slice_id}.h5", f"_{slice_i_str}.h5")
                img = self.get_image(img_path)
                img = np.expand_dims(img, -1)
                image = np.concatenate((image, img), axis=-1) if image is not None else img
        
        return image
    
    def get_patient_id_from_ct_path(self, ct_path):
        """Extract patient ID from CT slice path"""
        # Path format: /path/to/ct/patient_XXX/patient_XXX_NNN.h5
        parent_dir = os.path.basename(os.path.dirname(ct_path))
        return parent_dir
    
    def get_xray_path(self, ct_path, xray_type):
        """
        Get X-ray path from CT slice path.
        
        This is the KEY FIX for custom data structure.
        """
        patient_id = self.get_patient_id_from_ct_path(ct_path)
        
        # Check cache first
        if patient_id in self.xray_cache:
            return self.xray_cache[patient_id].get(xray_type)
        
        # Fallback: construct path manually
        xray_dir = Path(self.xray_base_dir) / patient_id
        
        if xray_type == 'PA':
            patterns = [
                f"{patient_id}_pa_drr_flipped.png",
                f"{patient_id}_pa_drr.png",
                "*_pa_drr_flipped.png",
                "*_pa_drr*.png"
            ]
        else:  # Lateral
            patterns = [
                f"{patient_id}_lat_drr_flipped.png",
                f"{patient_id}_lat_drr.png",
                "*_lat_drr_flipped.png",
                "*_lat_drr*.png"
            ]
        
        for pattern in patterns:
            matches = list(xray_dir.glob(pattern))
            if matches:
                return str(matches[0])
        
        # Last resort: use original path pattern with _xray1/_xray2
        if xray_type == 'PA':
            return str(xray_dir / f"{patient_id}_xray1_flipped.png")
        else:
            return str(xray_dir / f"{patient_id}_xray2_flipped.png")
    
    def __getitem__(self, i):
        example = dict()
        
        for index, input_type in enumerate(self.input_types):
            main_image_path = self.labels["file_path_"][i]
            
            if index == 0:  # ctslice
                image = self.get_ctslice(main_image_path)
                example[input_type] = self.dict_preprocessing[input_type](image)
            else:
                if input_type in ['PA', 'Lateral']:
                    # Use custom path resolution
                    image_path = self.get_xray_path(main_image_path, input_type)
                    
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(
                            f"X-ray not found: {image_path}\n"
                            f"CT path: {main_image_path}\n"
                            f"Expected type: {input_type}"
                        )
                    
                    image = self.get_image(image_path)
                    image, src_campose = self.apply_preprocessing_xray_according2cam(image, input_type)
                    image = np.expand_dims(image, -1)
                    image = np.concatenate((image, image, image), axis=-1)
                    example[input_type] = self.dict_preprocessing[input_type](image)
                    example[f"{input_type}_cam"] = src_campose
        
        for k in self.labels:
            example[k] = self.labels[k][i]
        
        return example


# Compatibility wrapper classes for LIDC-style imports
class Custom256Train(Dataset):
    """Training dataset wrapper"""
    
    def __init__(self, size, training_images_list_file, dataset_class, opt):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        
        # Always use Custom256MultiInputDataset
        num_ctslice_per_item = opt.get('num_ctslice_per_item', 1)
        self.data = Custom256MultiInputDataset(
            paths=paths, 
            opt=opt, 
            size=size, 
            num_ctslice_per_item=num_ctslice_per_item
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]


class Custom256Test(Dataset):
    """Test/Validation dataset wrapper"""
    
    def __init__(self, size, test_images_list_file, dataset_class, opt):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        
        num_ctslice_per_item = opt.get('num_ctslice_per_item', 1)
        self.data = Custom256MultiInputDataset(
            paths=paths, 
            opt=opt, 
            size=size, 
            num_ctslice_per_item=num_ctslice_per_item
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
