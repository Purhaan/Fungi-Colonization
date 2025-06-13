import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any

class MycorrhizalCNN(nn.Module):
    """CNN model for mycorrhizal colonization detection."""
    
    def __init__(self, model_type: str = "ResNet18", num_classes: int = 5, 
                 pretrained: bool = True):
        super(MycorrhizalCNN, self).__init__()
        
        self.model_type = model_type
        self.num_classes = num_classes
        
        if model_type == "ResNet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif model_type == "ResNet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif model_type == "EfficientNetB0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.backbone.classifier = nn.Linear(
                self.backbone.classifier[1].in_features, num_classes
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Feature extraction hook for Grad-CAM
        self.feature_maps = None
        self.register_hooks()
    
    def register_hooks(self):
        """Register hooks for feature extraction."""
        if self.model_type.startswith("ResNet"):
            self.backbone.layer4.register_forward_hook(self.save_feature_maps)
        elif self.model_type == "EfficientNetB0":
            self.backbone.features.register_forward_hook(self.save_feature_maps)
    
    def save_feature_maps(self, module, input, output):
        """Hook function to save feature maps."""
        self.feature_maps = output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.model_type.startswith("ResNet"):
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.backbone.fc(x)
        else:
            x = self.backbone(x)
        
        return x
    
    def get_feature_maps(self) -> torch.Tensor:
        """Get feature maps for Grad-CAM."""
        return self.feature_maps
