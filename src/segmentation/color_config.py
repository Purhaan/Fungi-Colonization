#!/usr/bin/env python3
"""
Color configuration for mycorrhizal structure segmentation
"""

# Color mapping for different fungal structures
STRUCTURE_COLORS = {
    "background": {"color": "#000000", "rgb": (0, 0, 0), "label": 0},
    "arbuscules": {"color": "#FF0000", "rgb": (255, 0, 0), "label": 1},
    "vesicles": {"color": "#00FF00", "rgb": (0, 255, 0), "label": 2}, 
    "hyphae": {"color": "#0000FF", "rgb": (0, 0, 255), "label": 3},
    "spores": {"color": "#FFFF00", "rgb": (255, 255, 0), "label": 4},
    "entry_points": {"color": "#FF00FF", "rgb": (255, 0, 255), "label": 5},
    "root_tissue": {"color": "#808080", "rgb": (128, 128, 128), "label": 6}
}

def get_structure_names():
    """Get list of structure names"""
    return list(STRUCTURE_COLORS.keys())

def get_color_from_label(label):
    """Get color info from label index"""
    for name, info in STRUCTURE_COLORS.items():
        if info["label"] == label:
            return info
    return None

def rgb_to_label(rgb_array):
    """Convert RGB values to class labels"""
    import numpy as np
    label_mask = np.zeros(rgb_array.shape[:2], dtype=np.uint8)
    
    for structure, info in STRUCTURE_COLORS.items():
        # Find pixels matching this color (with tolerance)
        matches = np.all(np.abs(rgb_array - info["rgb"]) <= 10, axis=2)
        label_mask[matches] = info["label"]
    
    return label_mask
