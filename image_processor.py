import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, List, Union

class ImageProcessor:
    """Handles image preprocessing and augmentation for mycorrhizal detection."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.augmentation_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path: str, augment: bool = False) -> torch.Tensor:
        """Preprocess image for model input."""
        image = Image.open(image_path).convert('RGB')
        
        if augment:
            return self.augmentation_transform(image)
        else:
            return self.transform(image)
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using bilateral filtering."""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def segment_roi(self, image: np.ndarray) -> np.ndarray:
        """Segment region of interest (root area) from background."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to create binary mask
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        result = image.copy()
        result[mask == 0] = 0
        
        return result
    
    def create_grid_overlay(self, image: np.ndarray, grid_size: int = 50) -> np.ndarray:
        """Create grid overlay for intersection counting method."""
        h, w = image.shape[:2]
        overlay = image.copy()
        
        # Draw vertical lines
        for x in range(0, w, grid_size):
            cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)
        
        # Draw horizontal lines
        for y in range(0, h, grid_size):
            cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
        
        return overlay
    
    def extract_patches(self, image: np.ndarray, patch_size: int = 64, 
                       stride: int = 32) -> List[np.ndarray]:
        """Extract overlapping patches from image for detailed analysis."""
        h, w = image.shape[:2]
        patches = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
        
        return patches
    
    def normalize_staining(self, image: np.ndarray) -> np.ndarray:
        """Normalize staining variations across images."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Normalize each channel
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
        b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
        
        # Merge and convert back
        normalized = cv2.merge([l, a, b])
        return cv2.cvtColor(normalized, cv2.COLOR_LAB2RGB)
