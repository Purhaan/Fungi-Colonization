#!/usr/bin/env python3
"""
Fixed Trainer for segmentation models - corrected import issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# Fixed import structure
try:
    from .models import UNet, DeepLabV3Segmentation
    from .dataset import SegmentationDataset
    from .color_config import STRUCTURE_COLORS
except ImportError:
    # Fallback for standalone usage
    from models import UNet, DeepLabV3Segmentation
    from dataset import SegmentationDataset
    from color_config import STRUCTURE_COLORS

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
        
        # Loss function with class weighting for imbalanced data
        self.criterion = nn.CrossEntropyLoss(weight=self._calculate_class_weights())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced datasets"""
        # Default weights - should be calculated from actual data
        weights = torch.ones(self.num_classes)
        weights[0] = 0.1  # Background typically dominates
        return weights.to(self.device)
    
    def _create_model(self):
        """Create segmentation model based on architecture"""
        if self.model_architecture == "U-Net":
            return UNet(num_classes=self.num_classes)
        elif self.model_architecture == "DeepLabV3":
            return DeepLabV3Segmentation(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {self.model_architecture}")
    
    def prepare_data(self, approved_datasets, val_split=0.2):
        """Prepare training and validation datasets with better error handling"""
        try:
            # Create dataset
            full_dataset = SegmentationDataset(approved_datasets)
            
            if len(full_dataset) == 0:
                raise ValueError("No valid datasets found")
            
            # Split into train and validation
            train_size = max(1, int((1 - val_split) * len(full_dataset)))
            val_size = len(full_dataset) - train_size
            
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )
            
            # Create data loaders with better settings
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=min(self.batch_size, len(self.train_dataset)), 
                shuffle=True,
                num_workers=0,  # For compatibility
                pin_memory=torch.cuda.is_available(),
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                self.val_dataset, 
                batch_size=min(self.batch_size, len(self.val_dataset)), 
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
            
            print(f"Dataset prepared: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
            
        except Exception as e:
            raise RuntimeError(f"Failed to prepare data: {str(e)}")
    
    def train_epoch(self):
        """Train for one epoch with improved error handling"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        try:
            for batch_idx, (images, masks) in enumerate(self.train_loader):
                # Move to device
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                # Ensure correct dimensions
                if len(images.shape) == 3:
                    images = images.unsqueeze(0)
                if len(masks.shape) == 2:
                    masks = masks.unsqueeze(0)
                
                # Forward pass
                outputs = self.model(images)
                
                # Resize outputs to match mask size if needed
                if outputs.shape[2:] != masks.shape[1:]:
                    outputs = F.interpolate(
                        outputs, 
                        size=masks.shape[1:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU out of memory. Try reducing batch size from {self.batch_size}")
                torch.cuda.empty_cache()
            raise e
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self):
        """Validate for one epoch with metrics calculation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct_pixels = 0
        total_pixels = 0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                # Ensure correct dimensions
                if len(images.shape) == 3:
                    images = images.unsqueeze(0)
                if len(masks.shape) == 2:
                    masks = masks.unsqueeze(0)
                
                outputs = self.model(images)
                
                # Resize outputs to match mask size
                if outputs.shape[2:] != masks.shape[1:]:
                    outputs = F.interpolate(
                        outputs, 
                        size=masks.shape[1:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=1)
                correct_pixels += (predictions == masks).sum().item()
                total_pixels += masks.numel()
        
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct_pixels / max(total_pixels, 1)
        
        self.val_losses.append(avg_loss)
        
        # Learning rate scheduling
        self.scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        
        return avg_loss, accuracy
    
    def save_model(self, path):
        """Save the trained model with comprehensive metadata"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_architecture': self.model_architecture,
                'num_classes': self.num_classes,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'structure_colors': STRUCTURE_COLORS,
                'training_completed': True
            }, path)
            print(f"Model saved successfully to {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {str(e)}")
    
    def load_model(self, path):
        """Load a trained model with error handling"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

# Additional utility functions for better training
def calculate_iou(pred, target, num_classes):
    """Calculate Intersection over Union for segmentation"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = 1.0  # Perfect score for classes not present
        else:
            iou = intersection / union
        
        ious.append(iou.item())
    
    return np.mean(ious)
