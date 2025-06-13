import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Optional

from .model import MycorrhizalCNN
from .image_processor import ImageProcessor

class GradCAMVisualizer:
    """Generates Grad-CAM visualizations for model explainability."""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = ImageProcessor()
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model_type = checkpoint.get('model_type', 'ResNet18')
        
        self.model = MycorrhizalCNN(model_type=model_type, num_classes=5)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Hook for gradients
        self.gradients = None
        self.register_hooks()
    
    def register_hooks(self):
        """Register hooks to capture gradients and activations."""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hook on the last convolutional layer
        if hasattr(self.model.backbone, 'layer4'):  # ResNet
            self.model.backbone.layer4.register_backward_hook(backward_hook)
        elif hasattr(self.model.backbone, 'features'):  # EfficientNet
            self.model.backbone.features.register_backward_hook(backward_hook)
    
    def generate_gradcam(self, image_path: str, target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap for an image."""
        # Preprocess image
        input_tensor = self.image_processor.preprocess_image(image_path)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        # Get feature maps and gradients
        feature_maps = self.model.get_feature_maps()
        gradients = self.gradients
        
        # Generate Grad-CAM
        gradcam = self._compute_gradcam(feature_maps, gradients)
        
        # Resize to original image size
        original_image = Image.open(image_path).convert('RGB')
        gradcam_resized = cv2.resize(gradcam, original_image.size)
        
        # Create visualization
        visualization = self._create_visualization(original_image, gradcam_resized)
        
        return visualization
    
    def _compute_gradcam(self, feature_maps: torch.Tensor, 
                        gradients: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM from feature maps and gradients."""
        # Pool gradients across spatial dimensions
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight feature maps by gradients
        for i in range(feature_maps.size()[1]):
            feature_maps[0, i, :, :] *= pooled_gradients[i]
        
        # Average across feature map channels
        gradcam = torch.mean(feature_maps, dim=1).squeeze()
        
        # Apply ReLU
        gradcam = F.relu(gradcam)
        
        # Normalize to 0-1
        gradcam -= gradcam.min()
        gradcam /= gradcam.max()
        
        return gradcam.cpu().detach().numpy()
    
    def _create_visualization(self, original_image: Image.Image, 
                            gradcam: np.ndarray) -> np.ndarray:
        """Create visualization combining original image and Grad-CAM."""
        # Convert PIL image to numpy array
        img_array = np.array(original_image)
        
        # Apply colormap to Grad-CAM
        colormap = cm.get_cmap('jet')
        heatmap = colormap(gradcam)
        heatmap = np.uint8(255 * heatmap[:, :, :3])  # Remove alpha channel
        
        # Blend original image with heatmap
        overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
        
        return overlay
    
    def generate_class_specific_gradcam(self, image_path: str) -> dict:
        """Generate Grad-CAM for all classes."""
        class_names = [
            "Not colonized",
            "Lightly colonized", 
            "Moderately colonized",
            "Heavily colonized",
            "Not annotated"
        ]
        
        gradcams = {}
        
        for class_idx, class_name in enumerate(class_names):
            try:
                gradcam = self.generate_gradcam(image_path, target_class=class_idx)
                gradcams[class_name] = gradcam
            except Exception as e:
                print(f"Error generating Grad-CAM for {class_name}: {e}")
        
        return gradcams
    
    def create_comparison_plot(self, image_path: str, gradcams: dict) -> np.ndarray:
        """Create a comparison plot showing original image and Grad-CAMs."""
        original_image = Image.open(image_path).convert('RGB')
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Show Grad-CAMs for each class
        class_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for idx, (class_name, gradcam) in enumerate(gradcams.items()):
            if idx < len(class_positions):
                row, col = class_positions[idx]
                axes[row, col].imshow(gradcam)
                axes[row, col].set_title(f"Grad-CAM: {class_name}")
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image array
        fig.canvas.draw()
        plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return plot_array
    
    def analyze_attention_regions(self, image_path: str, 
                                gradcam: np.ndarray) -> dict:
        """Analyze which regions the model is focusing on."""
        # Threshold heatmap to find high attention regions
        threshold = 0.5
        high_attention = gradcam > threshold
        
        # Find connected components
        labeled_regions, num_regions = ndimage.label(high_attention)
        
        # Analyze each region
        regions = []
        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            region_area = np.sum(region_mask)
            
            # Find center of mass
            center_y, center_x = ndimage.center_of_mass(region_mask)
            
            # Calculate average attention value
            avg_attention = np.mean(gradcam[region_mask])
            
            regions.append({
                'region_id': region_id,
                'area': region_area,
                'center': (int(center_x), int(center_y)),
                'avg_attention': avg_attention
            })
        
        # Sort by attention value
        regions.sort(key=lambda x: x['avg_attention'], reverse=True)
        
        return {
            'num_attention_regions': num_regions,
            'total_attention_area': np.sum(high_attention),
            'attention_percentage': np.sum(high_attention) / gradcam.size * 100,
            'top_regions': regions[:5]  # Top 5 regions
        }
