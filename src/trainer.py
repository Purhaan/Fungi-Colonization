import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from .model import MycorrhizalCNN
from .image_processor import ImageProcessor

class MycorrhizalDataset(Dataset):
    """Dataset class for mycorrhizal colonization images."""
    
    def __init__(self, image_dir: str, annotation_dir: str, 
                 image_processor: ImageProcessor, augment: bool = False):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_processor = image_processor
        self.augment = augment
        
        self.data = self._load_data()
        
        # Map annotation types to class indices
        self.class_mapping = {
            "Not colonized": 0,
            "Lightly colonized": 1,
            "Moderately colonized": 2,
            "Heavily colonized": 3,
            "Not annotated": 4
        }
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load image and annotation data."""
        data = []
        
        annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.json')]
        
        for annotation_file in annotation_files:
            with open(os.path.join(self.annotation_dir, annotation_file), 'r') as f:
                annotation = json.load(f)
            
            image_path = os.path.join(self.image_dir, annotation['image'])
            if os.path.exists(image_path):
                data.append({
                    'image_path': image_path,
                    'annotation': annotation
                })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.data[idx]
        
        # Load and preprocess image
        image = self.image_processor.preprocess_image(
            item['image_path'], augment=self.augment
        )
        
        # Get class label
        annotation_type = item['annotation']['annotation_type']
        label = self.class_mapping.get(annotation_type, 4)  # Default to "Not annotated"
        
        return image, label

class ModelTrainer:
    """Handles model training and validation."""
    
    def __init__(self, model_type: str = "ResNet18", learning_rate: float = 0.001,
                 batch_size: int = 16, use_gpu: bool = True):
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
        # Initialize model
        self.model = MycorrhizalCNN(model_type=model_type, num_classes=5)
        self.model.to(self.device)
        
        # Initialize image processor
        self.image_processor = ImageProcessor()
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def prepare_data(self, image_dir: str, annotation_dir: str) -> None:
        """Prepare training and validation datasets."""
        # Create full dataset
        full_dataset = MycorrhizalDataset(
            image_dir, annotation_dir, self.image_processor, augment=True
        )
        
        # Split into train and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        # Validation
        val_loss, val_acc = self.validate()
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        
        # Update learning rate
        self.scheduler.step()
        
        return epoch_loss
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100 * correct / total
        
        return val_loss, val_acc
    
    def save_model(self, path: str) -> None:
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
    
    def get_training_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics for visualization."""
        return {
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_accuracy': self.train_accuracies,
            'val_accuracy': self.val_accuracies
        }
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Classification report
        class_names = ["Not colonized", "Lightly colonized", "Moderately colonized", 
                      "Heavily colonized", "Not annotated"]
        
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names
        }
# ADD this class to the END of src/trainer.py:

class ProgressiveTrainer(ModelTrainer):
    """Progressive training with transfer learning for better results"""
    
    def __init__(self, model_type: str = "ResNet18", learning_rate: float = 0.001,
                 batch_size: int = 16, use_gpu: bool = True):
        super().__init__(model_type, learning_rate, batch_size, use_gpu)
        
        # Use improved model with transfer learning
        self.model = MycorrhizalCNN(
            model_type=model_type, 
            num_classes=5, 
            pretrained=True,
            freeze_backbone=True  # Start with frozen backbone
        )
        self.model.to(self.device)
        
        # Progressive learning schedule
        self.training_phases = [
            {'name': 'Phase 1: Head Only', 'epochs': 10, 'lr': 1e-3, 'freeze': True},
            {'name': 'Phase 2: Fine-tune Last Layers', 'epochs': 15, 'lr': 1e-4, 'freeze': False},
            {'name': 'Phase 3: Full Fine-tune', 'epochs': 10, 'lr': 1e-5, 'freeze': False}
        ]
        
    def progressive_train(self, image_dir: str, annotation_dir: str):
        """Train progressively with different learning rates and frozen layers"""
        
        # Prepare data once
        self.prepare_data(image_dir, annotation_dir)
        
        all_metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 
                      'train_accuracy': [], 'val_accuracy': [], 'phase': []}
        
        total_epochs = 0
        
        for phase_idx, phase in enumerate(self.training_phases):
            st.subheader(f"🎯 {phase['name']}")
            st.write(f"Epochs: {phase['epochs']}, Learning Rate: {phase['lr']}")
            
            # Adjust model for this phase
            if phase_idx == 1:  # Phase 2: unfreeze last layers
                self.model.unfreeze_backbone(layers_to_unfreeze=1)
                st.info("🔓 Unfroze last backbone layer")
            elif phase_idx == 2:  # Phase 3: unfreeze all
                self.model.unfreeze_backbone(layers_to_unfreeze=4)
                st.info("🔓 Unfroze all backbone layers")
            
            # Update optimizer for new learning rate
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=phase['lr']
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=phase['epochs']//2, gamma=0.5
            )
            
            # Train for this phase
            phase_progress = st.progress(0)
            for epoch in range(phase['epochs']):
                loss = self.train_epoch()
                
                # Update progress
                phase_progress.progress((epoch + 1) / phase['epochs'])
                
                # Store metrics
                all_metrics['epoch'].append(total_epochs + epoch + 1)
                all_metrics['train_loss'].append(self.train_losses[-1])
                all_metrics['val_loss'].append(self.val_losses[-1])
                all_metrics['train_accuracy'].append(self.train_accuracies[-1])
                all_metrics['val_accuracy'].append(self.val_accuracies[-1])
                all_metrics['phase'].append(phase['name'])
                
                if epoch % 5 == 0:
                    st.write(f"  Epoch {epoch+1}/{phase['epochs']}: Loss = {loss:.4f}, "
                           f"Val Acc = {self.val_accuracies[-1]:.1f}%")
            
            total_epochs += phase['epochs']
            st.success(f"✅ {phase['name']} complete!")
        
        # Plot progressive training results
        self.plot_progressive_results(all_metrics)
        
        return self.model
    
    def plot_progressive_results(self, metrics):
        """Plot results across all training phases"""
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        
        df = pd.DataFrame(metrics)
        
        # Create subplot with phase annotations
        fig = go.Figure()
        
        # Add training and validation loss
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_loss'], 
                               name='Training Loss', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'], 
                               name='Validation Loss', line=dict(color='red')))
        
        # Add phase boundaries
        phase_changes = [0]
        current_phase = df['phase'].iloc[0]
        for i, phase in enumerate(df['phase']):
            if phase != current_phase:
                phase_changes.append(i)
                current_phase = phase
        phase_changes.append(len(df))
        
        for i, change_point in enumerate(phase_changes[1:]):
            fig.add_vline(x=df['epoch'].iloc[change_point-1], 
                         line_dash="dash", line_color="gray",
                         annotation_text=f"Phase {i+2}")
        
        fig.update_layout(title="Progressive Training Results",
                         xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)
        
        # Accuracy plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['epoch'], y=df['train_accuracy'], 
                                name='Training Accuracy', line=dict(color='green')))
        fig2.add_trace(go.Scatter(x=df['epoch'], y=df['val_accuracy'], 
                                name='Validation Accuracy', line=dict(color='orange')))
        
        for i, change_point in enumerate(phase_changes[1:]):
            fig2.add_vline(x=df['epoch'].iloc[change_point-1], 
                          line_dash="dash", line_color="gray")
        
        fig2.update_layout(title="Progressive Training Accuracy",
                          xaxis_title="Epoch", yaxis_title="Accuracy (%)")
        st.plotly_chart(fig2, use_container_width=True)

# MODIFY the train_model_page() function in app.py to include progressive training option:
# ADD this after the existing training button:

if st.button("🚀 Progressive Training (Better Results)", type="primary"):
    # Use progressive trainer instead
    trainer = ProgressiveTrainer(
        model_type=model_type,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_gpu=use_gpu
    )
    
    # Run progressive training
    model = trainer.progressive_train("data/raw", "data/annotations")
    
    # Save model same as before
    # ... (rest of saving code remains the same)
