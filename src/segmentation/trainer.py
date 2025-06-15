#!/usr/bin/env python3
"""
Trainer for segmentation models
"""

import torch.nn.functional as Fimport torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from .models import UNet, DeepLabV3Segmentation
from .dataset import SegmentationDataset
from .color_config import STRUCTURE_COLORS

class SegmentationTrainer:
    """Trainer for mycorrhizal segmentation models"""
    
    def __init__(self, model_architecture="U-Net", num_classes=7, 
                 learning_rate=0.001, batch_size=4, device=None):
        self.model_architecture = model_architecture
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def _create_model(self):
        """Create segmentation model based on architecture"""
        if self.model_architecture == "U-Net":
            return UNet(num_classes=self.num_classes)
        elif self.model_architecture == "DeepLabV3":
            return DeepLabV3Segmentation(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {self.model_architecture}")
    
    def prepare_data(self, approved_datasets, val_split=0.2):
        """Prepare training and validation datasets"""
        # Create dataset
        full_dataset = SegmentationDataset(approved_datasets)
        
        # Split into train and validation
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Resize outputs to match mask size if needed
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
            
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                # Resize outputs to match mask size if needed
                if outputs.shape[2:] != masks.shape[1:]:
                    outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': self.model_architecture,
            'num_classes': self.num_classes,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'structure_colors': STRUCTURE_COLORS
        }, path)
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
