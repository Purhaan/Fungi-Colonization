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

class MycorrhizalSegmentationModel(nn.Module):
    """U-Net style model for segmentation of mycorrhizal structures."""
    
    def __init__(self, num_classes: int = 3):  # background, colonized, non-colonized
        super(MycorrhizalSegmentationModel, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final classification layer
        self.final = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)
