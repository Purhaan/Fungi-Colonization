#!/usr/bin/env python3
"""
Dataset class for color-coded segmentation training
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import json
from .color_config import STRUCTURE_COLORS, rgb_to_label

class SegmentationDataset(Dataset):
    """Dataset for mycorrhizal segmentation with color-coded masks"""
    
    def __init__(self, metadata_list, transform=None, augment=False):
        self.metadata_list = metadata_list
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        return len(self.metadata_list)
    
    def __getitem__(self, idx):
        metadata = self.metadata_list[idx]
        
        # Load original image
        image = Image.open(metadata['original_path']).convert('RGB')
        
        # Load annotation mask
        mask = Image.open(metadata['mask_path']).convert('RGB')
        
        # Ensure same size
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        
        # Convert to numpy arrays
        image_array = np.array(image)
        mask_array = np.array(mask)
        
        # Convert color mask to label mask
        label_mask = rgb_to_label(mask_array)
        
        # Apply transforms
        if self.transform:
            image_array = self.transform(image_array)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_array).float()
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        
        label_tensor = torch.from_numpy(label_mask).long()
        
        return image_tensor, label_tensor
