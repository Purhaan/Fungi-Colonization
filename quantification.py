import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any
import json
from scipy import ndimage
from skimage import measure, morphology

class ColonizationQuantifier:
    """Quantifies mycorrhizal colonization using various methods."""
    
    def __init__(self, grid_size: int = 50):
        self.grid_size = grid_size
    
    def quantify_colonization(self, image_path: str, 
                            prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify colonization using multiple methods."""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply different quantification methods
        grid_method = self._gridline_intersection_method(image, prediction)
        area_method = self._area_percentage_method(image, prediction)
        intensity_method = self._intensity_analysis_method(image, prediction)
        
        # Combine results
        final_percentage = np.mean([
            grid_method['percentage'],
            area_method['percentage'],
            intensity_method['percentage']
        ])
        
        result = {
            'colonization_percentage': final_percentage,
            'grid_method': grid_method,
            'area_method': area_method,
            'intensity_method': intensity_method,
            'consensus_confidence': self._calculate_consensus_confidence([
                grid_method['percentage'],
                area_method['percentage'], 
                intensity_method['percentage']
            ])
        }
        
        return result
    
    def _gridline_intersection_method(self, image: np.ndarray, 
                                    prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify using gridline intersection method."""
        h, w = image.shape[:2]
        
        # Create grid
        grid_image = self._create_grid_overlay(image)
        
        # Detect colonized regions
        colonized_mask = self._detect_colonized_regions(image, prediction)
        
        # Count intersections
        total_intersections = 0
        colonized_intersections = 0
        
        # Vertical lines
        for x in range(0, w, self.grid_size):
            for y in range(0, h):
                if self._is_root_tissue(image, x, y):
                    total_intersections += 1
                    if colonized_mask[y, x] > 0:
                        colonized_intersections += 1
        
        # Horizontal lines
        for y in range(0, h, self.grid_size):
            for x in range(0, w):
                if self._is_root_tissue(image, x, y):
                    total_intersections += 1
                    if colonized_mask[y, x] > 0:
                        colonized_intersections += 1
        
        percentage = (colonized_intersections / total_intersections * 100) if total_intersections > 0 else 0
        
        return {
            'percentage': percentage,
            'total_intersections': total_intersections,
            'colonized_intersections': colonized_intersections,
            'method': 'gridline_intersection'
        }
    
    def _area_percentage_method(self, image: np.ndarray, 
                              prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify using area percentage method."""
        # Segment root tissue
        root_mask = self._segment_root_tissue(image)
        
        # Detect colonized regions
        colonized_mask = self._detect_colonized_regions(image, prediction)
        
        # Calculate areas
        total_root_area = np.sum(root_mask > 0)
        colonized_area = np.sum((root_mask > 0) & (colonized_mask > 0))
        
        percentage = (colonized_area / total_root_area * 100) if total_root_area > 0 else 0
        
        return {
            'percentage': percentage,
            'total_root_area': total_root_area,
            'colonized_area': colonized_area,
            'method': 'area_percentage'
        }
    
    def _intensity_analysis_method(self, image: np.ndarray, 
                                 prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify using intensity analysis method."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Segment root tissue
        root_mask = self._segment_root_tissue(image)
        
        # Analyze intensity distribution in root regions
        root_intensities = gray[root_mask > 0]
        
        # Define thresholds for colonized regions (typically darker)
        # This is simplified - in practice, you'd train on labeled data
        threshold = np.percentile(root_intensities, 30)  # Bottom 30% intensity
        
        colonized_pixels = np.sum((gray < threshold) & (root_mask > 0))
        total_root_pixels = np.sum(root_mask > 0)
        
        percentage = (colonized_pixels / total_root_pixels * 100) if total_root_pixels > 0 else 0
        
        return {
            'percentage': percentage,
            'total_root_pixels': total_root_pixels,
            'colonized_pixels': colonized_pixels,
            'intensity_threshold': threshold,
            'method': 'intensity_analysis'
        }
    
    def _create_grid_overlay(self, image: np.ndarray) -> np.ndarray:
        """Create grid overlay for visualization."""
        h, w = image.shape[:2]
        overlay = image.copy()
        
        # Draw vertical lines
        for x in range(0, w, self.grid_size):
            cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)
        
        # Draw horizontal lines
        for y in range(0, h, self.grid_size):
            cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
        
        return overlay
    
    def _segment_root_tissue(self, image: np.ndarray) -> np.ndarray:
        """Segment root tissue from background."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold using Otsu's method
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _detect_colonized_regions(self, image: np.ndarray, 
                                prediction: Dict[str, Any]) -> np.ndarray:
        """Detect colonized regions based on prediction and image analysis."""
        # This is a simplified version - in practice, you'd use the trained model
        # to generate pixel-level predictions
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Use prediction confidence to adjust thresholds
        confidence = prediction.get('confidence', 0.5)
        
        # Detect dark regions (potential arbuscules/hyphae)
        dark_lower = np.array([0, 0, 0])
        dark_upper = np.array([180, 255, int(100 * confidence)])
        dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)
        
        # Detect specific color ranges based on detected features
        colonized_mask = dark_mask.copy()
        
        if 'vesicles' in prediction.get('detected_features', []):
            # Add vesicle-like regions (lighter, round)
            vesicle_lower = np.array([0, 0, 150])
            vesicle_upper = np.array([180, 100, 255])
            vesicle_mask = cv2.inRange(hsv, vesicle_lower, vesicle_upper)
            colonized_mask = cv2.bitwise_or(colonized_mask, vesicle_mask)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        colonized_mask = cv2.morphologyEx(colonized_mask, cv2.MORPH_CLOSE, kernel)
        
        return colonized_mask
    
    def _is_root_tissue(self, image: np.ndarray, x: int, y: int) -> bool:
        """Check if a point is within root tissue."""
        if x >= image.shape[1] or y >= image.shape[0]:
            return False
        
        # Simple check based on intensity
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray[y, x] > 30  # Exclude very dark background
    
    def _calculate_consensus_confidence(self, percentages: List[float]) -> float:
        """Calculate confidence based on agreement between methods."""
        # Calculate coefficient of variation
        mean_percentage = np.mean(percentages)
        std_percentage = np.std(percentages)
        
        if mean_percentage == 0:
            return 0.0
        
        cv = std_percentage / mean_percentage
        
        # Higher agreement (lower CV) = higher confidence
        confidence = max(0.0, 1.0 - cv)
        return confidence
    
    def generate_quantification_report(self, results: Dict[str, Any], 
                                     image_path: str) -> Dict[str, Any]:
        """Generate comprehensive quantification report."""
        report = {
            'image_path': image_path,
            'final_colonization_percentage': results['colonization_percentage'],
            'consensus_confidence': results['consensus_confidence'],
            'method_breakdown': {
                'gridline_intersection': results['grid_method']['percentage'],
                'area_percentage': results['area_method']['percentage'],
                'intensity_analysis': results['intensity_method']['percentage']
            },
            'quality_metrics': {
                'method_agreement': results['consensus_confidence'],
                'recommended_confidence': 'High' if results['consensus_confidence'] > 0.8 else 
                                        'Medium' if results['consensus_confidence'] > 0.6 else 'Low'
            }
        }
        
        return report
