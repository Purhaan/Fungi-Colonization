import torch
import numpy as np
from PIL import Image
import cv2
import os
from typing import Dict, List, Any, Tuple
import json

from .model import MycorrhizalCNN
from .image_processor import ImageProcessor

class ModelInference:
    """Handles model inference for new images."""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = ImageProcessor()
        
        try:
            # Load model with better error handling
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            model_type = checkpoint.get('model_type', 'ResNet18')
            
            self.model = MycorrhizalCNN(model_type=model_type, num_classes=5)
            
            # Check if state dict is compatible
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except KeyError:
                raise ValueError("Invalid model checkpoint - missing model_state_dict")
            except RuntimeError as e:
                raise ValueError(f"Model architecture mismatch: {e}")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
        
        # Class mappings
        self.class_names = [
            "Not colonized",
            "Lightly colonized", 
            "Moderately colonized",
            "Heavily colonized",
            "Not annotated"
        ]
        
        # Colonization percentages for each class (approximate)
        self.class_percentages = {
            0: 0,     # Not colonized
            1: 25,    # Lightly colonized
            2: 50,    # Moderately colonized
            3: 80,    # Heavily colonized
            4: 0      # Not annotated
        }
    
    def predict(self, image_path: str, confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Predict colonization for a single image."""
        # Preprocess image
        image_tensor = self.image_processor.preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_class = predicted_class.item()
        
        # Get prediction results
        class_name = self.class_names[predicted_class]
        estimated_percentage = self.class_percentages[predicted_class]
        
        # Detect specific features
        detected_features = self._detect_features(image_path, probabilities)
        
        result = {
            'predicted_class': predicted_class,
            'class_name': class_name,
            'confidence': confidence,
            'estimated_percentage': estimated_percentage,
            'all_probabilities': probabilities.cpu().numpy().tolist()[0],
            'detected_features': detected_features,
            'above_threshold': confidence >= confidence_threshold
        }
        
        return result
    
    def _detect_features(self, image_path: str, probabilities: torch.Tensor) -> List[str]:
        """Detect specific mycorrhizal features."""
        features = []
        
        # Load original image for feature detection
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for different structures
        # Arbuscules (tree-like structures) - typically darker
        arbuscule_lower = np.array([0, 0, 0])
        arbuscule_upper = np.array([180, 255, 100])
        arbuscule_mask = cv2.inRange(hsv, arbuscule_lower, arbuscule_upper)
        
        # Vesicles (round structures) - typically lighter
        vesicle_lower = np.array([0, 0, 150])
        vesicle_upper = np.array([180, 100, 255])
        vesicle_mask = cv2.inRange(hsv, vesicle_lower, vesicle_upper)
        
        # Hyphae (thread-like structures) - intermediate
        hyphae_lower = np.array([0, 0, 80])
        hyphae_upper = np.array([180, 150, 200])
        hyphae_mask = cv2.inRange(hsv, hyphae_lower, hyphae_upper)
        
        # Count pixels for each feature type
        total_pixels = image.shape[0] * image.shape[1]
        
        arbuscule_ratio = np.sum(arbuscule_mask > 0) / total_pixels
        vesicle_ratio = np.sum(vesicle_mask > 0) / total_pixels
        hyphae_ratio = np.sum(hyphae_mask > 0) / total_pixels
        
        # Thresholds for feature detection
        if arbuscule_ratio > 0.05:
            features.append("arbuscules")
        if vesicle_ratio > 0.03:
            features.append("vesicles")
        if hyphae_ratio > 0.1:
            features.append("hyphae")
        
        return features
    
    def predict_batch(self, image_paths: List[str], 
                     confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Predict colonization for multiple images."""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path, confidence_threshold)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'predicted_class': None,
                    'confidence': 0.0
                })
        
        return results
    
    def visualize_prediction(self, image_path: str, 
                           prediction: Dict[str, Any]) -> np.ndarray:
        """Create visualization of prediction results."""
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create overlay with prediction information
        overlay = image.copy()
        
        # Add text with prediction
        text = f"Class: {prediction['class_name']}"
        confidence_text = f"Confidence: {prediction['confidence']:.2f}"
        percentage_text = f"Est. Colonization: {prediction['estimated_percentage']}%"
        
        # Add text to image
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        cv2.putText(overlay, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        cv2.putText(overlay, percentage_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Add detected features
        if prediction['detected_features']:
            features_text = f"Features: {', '.join(prediction['detected_features'])}"
            cv2.putText(overlay, features_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # Add confidence color coding
        if prediction['confidence'] > 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif prediction['confidence'] > 0.6:
            color = (255, 255, 0)  # Yellow for medium confidence
        else:
            color = (255, 0, 0)  # Red for low confidence
        
        # Add colored border
        cv2.rectangle(overlay, (0, 0), (image.shape[1]-1, image.shape[0]-1), 
                     color, 5)
        
        return overlay
