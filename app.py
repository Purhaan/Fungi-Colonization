import streamlit as st
import os
import time
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image
import json
import zipfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from src.active_learning import ActiveLearningSelector, calculate_annotation_priority_score
from src.image_processor import ImageProcessor
from src.model import MycorrhizalCNN
from src.trainer import ModelTrainer
from src.inference import ModelInference
from src.quantification import ColonizationQuantifier
from src.gradcam import GradCAMVisualizer

# Page config
st.set_page_config(
    page_title="Mycorrhizal Colonization Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/annotations", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/results", exist_ok=True)
os.makedirs("models", exist_ok=True)

def main():
    st.title("🔬 Mycorrhizal Colonization Detection System")
    st.markdown("### AI-powered analysis of plant root microscope images")
    
    # Show system status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ PyTorch Available")
    with col2:
        st.success("✅ AI Features Active")
    with col3:
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.info(f"🖥️ Running on {device}")
    
    # Initialize session state
    if 'annotation_data' not in st.session_state:
        st.session_state.annotation_data = {}
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Upload & Annotate", "Train AI Model", "Batch Analysis", "Results & Export", "Model Explainability"]
    )
    
    if page == "Upload & Annotate":
        upload_and_annotate_page()
    elif page == "Train AI Model":
        train_model_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Results & Export":
        results_export_page()
    elif page == "Model Explainability":
        explainability_page()

def upload_and_annotate_page():
    st.header("📤 Upload Images & Manual Annotation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Microscope Images")
        
        # Optimized file uploader with better performance
        uploaded_files = st.file_uploader(
            "Choose microscope images of plant roots",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            help="Tip: Upload smaller images (< 5MB) for faster processing"
        )
        
        if uploaded_files:
            # Add progress bar for uploads
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Optimize image size for faster processing
                try:
                    image = Image.open(uploaded_file)
                    
                    # Resize large images to max 1024px while maintaining aspect ratio
                    max_size = (1024, 1024)
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                        st.info(f"📏 Resized {uploaded_file.name} for optimal performance")
                    
                    # Save optimized image
                    file_path = os.path.join("data/raw", uploaded_file.name)
                    
                    # Convert to RGB if necessary
                    if image.mode in ('RGBA', 'LA', 'P'):
                        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P':
                            image = image.convert('RGBA')
                        rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                        image = rgb_image
                    
                    # Save with optimization
                    image.save(file_path, format='JPEG', quality=85, optimize=True)
                    
                    st.success(f"✅ Uploaded: {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ All uploads complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
    
    with col2:
        st.subheader("Manual Annotation")
        
        # List available images with refresh button
        col2a, col2b = st.columns([3, 1])
        with col2b:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        image_files = []
        if os.path.exists("data/raw"):
            image_files = [f for f in os.listdir("data/raw") 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        
        if image_files:
            selected_image = st.selectbox("Select image to annotate:", image_files)
            
            if selected_image:
                image_path = os.path.join("data/raw", selected_image)
                
                try:
                    image = Image.open(image_path)
                    
                    # Display image with better sizing
                    st.image(image, caption=selected_image, use_column_width=True)
                    
                    # Check if annotation already exists
                    annotation_file = os.path.join("data/annotations", f"{selected_image}_annotation.json")
                    existing_annotation = None
                    
                    if os.path.exists(annotation_file):
                        try:
                            with open(annotation_file, 'r') as f:
                                existing_annotation = json.load(f)
                            st.info("📝 Previous annotation found - editing existing")
                        except:
                            pass
                    
                    # Annotation interface with existing values
                    st.markdown("**Annotation Guidelines:**")
                    st.markdown("- Heavily colonized: >70% root area")
                    st.markdown("- Moderately colonized: 30-70%")
                    st.markdown("- Lightly colonized: 10-30%")
                    st.markdown("- Not colonized: <10%")
                    
                    # Pre-fill with existing values if available
                    default_type = existing_annotation.get('annotation_type', "Not annotated") if existing_annotation else "Not annotated"
                    default_percentage = existing_annotation.get('colonization_percentage', 0) if existing_annotation else 0
                    default_features = existing_annotation.get('detected_features', []) if existing_annotation else []
                    
                    annotation_type = st.selectbox(
                        "Annotation type for this image:",
                        ["Not annotated", "Heavily colonized", "Moderately colonized", "Lightly colonized", "Not colonized"],
                        index=["Not annotated", "Heavily colonized", "Moderately colonized", "Lightly colonized", "Not colonized"].index(default_type)
                    )
                    
                    colonization_percentage = st.slider(
                        "Estimated colonization percentage:",
                        0, 100, default_percentage
                    )
                    
                    detected_features = st.multiselect(
                        "Manually detected features:",
                        ["Arbuscules", "Vesicles", "Hyphae", "Spores", "Entry points"],
                        default=default_features
                    )
                    
                    notes = st.text_area(
                        "Additional notes (optional):",
                        value=existing_annotation.get('notes', '') if existing_annotation else ''
                    )
                    
                    if st.button("💾 Save Annotation", type="primary"):
                        annotation_data = {
                            "image": selected_image,
                            "annotation_type": annotation_type,
                            "colonization_percentage": colonization_percentage,
                            "detected_features": detected_features,
                            "notes": notes,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        try:
                            with open(annotation_file, 'w') as f:
                                json.dump(annotation_data, f, indent=2)
                            
                            st.success("✅ Annotation saved successfully!")
                            st.session_state.annotation_data[selected_image] = annotation_data
                            
                            # Auto-refresh to show updated annotation
                            time.sleep(1)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error saving annotation: {str(e)}")
                
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
        else:
            st.info("👆 Upload images first to start annotating")
        
        # Display existing annotations section
        st.markdown("---")
        st.subheader("📋 Saved Annotations")
        
        annotation_files = []
        if os.path.exists("data/annotations"):
            annotation_files = [f for f in os.listdir("data/annotations") if f.endswith('.json')]
        
        if annotation_files:
            st.write(f"**Total annotations:** {len(annotation_files)}")
            
            # Load and display annotations
            annotations = []
            for ann_file in annotation_files:
                try:
                    with open(os.path.join("data/annotations", ann_file), 'r') as f:
                        data = json.load(f)
                        annotations.append(data)
                except:
                    continue
            
            if annotations:
                # Summary statistics
                df = pd.DataFrame(annotations)
                if 'colonization_percentage' in df.columns:
                    avg_colonization = df['colonization_percentage'].mean()
                    st.metric("Average Colonization", f"{avg_colonization:.1f}%")
                
                # Show recent annotations
                with st.expander(f"📝 View all {len(annotations)} annotations"):
                    for annotation in sorted(annotations, key=lambda x: x.get('timestamp', ''), reverse=True):
                        st.write(f"**{annotation['image']}** - {annotation['annotation_type']} ({annotation['colonization_percentage']}%)")
                        if annotation.get('detected_features'):
                            st.write(f"   Features: {', '.join(annotation['detected_features'])}")
                        if annotation.get('notes'):
                            st.write(f"   Notes: {annotation['notes']}")
                        st.write("---")
        else:
            st.info("No annotations saved yet")
            # ADD THIS FUNCTION to app.py (after upload_and_annotate_page function)

def smart_image_analysis(image_path):
    """Automatically analyze image quality and suggest colonization regions"""
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Image quality assessment
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Check blur (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check contrast
        contrast_score = gray.std()
        
        # Check brightness
        brightness_score = gray.mean()
        
        # 2. Automatic region detection
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Detect dark regions (potential colonization)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        dark_percentage = (np.sum(dark_mask > 0) / dark_mask.size) * 100
        
        # Detect medium-dark regions (potential structures)
        medium_mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 255, 150]))
        medium_percentage = (np.sum(medium_mask > 0) / medium_mask.size) * 100
        
        # Suggested colonization level based on automatic analysis
        if dark_percentage > 15:
            suggested_level = "Heavily colonized"
            suggested_percentage = min(80, dark_percentage * 4)
        elif dark_percentage > 8:
            suggested_level = "Moderately colonized" 
            suggested_percentage = min(60, dark_percentage * 5)
        elif dark_percentage > 3:
            suggested_level = "Lightly colonized"
            suggested_percentage = min(30, dark_percentage * 8)
        else:
            suggested_level = "Not colonized"
            suggested_percentage = max(5, dark_percentage * 2)
        
        return {
            'quality': {
                'blur_score': blur_score,
                'contrast_score': contrast_score, 
                'brightness_score': brightness_score,
                'quality_rating': 'Good' if blur_score > 100 and contrast_score > 40 else 'Poor'
            },
            'suggestions': {
                'level': suggested_level,
                'percentage': int(suggested_percentage),
                'dark_regions': dark_percentage,
                'confidence': min(0.9, dark_percentage / 20)
            },
            'detected_regions': {
                'dark_mask': dark_mask,
                'medium_mask': medium_mask
            }
        }
    except Exception as e:
        return None

# REPLACE the image display section in upload_and_annotate_page() with this:

if selected_image:
    image_path = os.path.join("data/raw", selected_image)
    
    try:
        image = Image.open(image_path)
        
        # Display image with better sizing
        st.image(image, caption=selected_image, use_column_width=True)
        
        # ADD SMART ANALYSIS
        if st.button("🤖 Smart Analysis", help="AI suggests colonization level"):
            with st.spinner("Analyzing image..."):
                analysis = smart_image_analysis(image_path)
                
                if analysis:
                    st.success("✅ Smart analysis complete!")
                    
                    # Quality assessment
                    quality = analysis['quality']
                    if quality['quality_rating'] == 'Good':
                        st.success(f"📸 Image Quality: {quality['quality_rating']}")
                    else:
                        st.warning(f"📸 Image Quality: {quality['quality_rating']} - Consider retaking")
                    
                    # AI suggestions
                    suggestions = analysis['suggestions']
                    st.info(f"🎯 **AI Suggestion:** {suggestions['level']} ({suggestions['percentage']}%)")
                    st.info(f"🎯 **Confidence:** {suggestions['confidence']:.1%}")
                    
                    # Show detected regions
                    col_a, col_b = st.columns(2)
                    with col_a:
                        dark_overlay = np.array(image).copy()
                        dark_overlay[analysis['detected_regions']['dark_mask'] > 0] = [255, 0, 0]
                        blended = cv2.addWeighted(np.array(image), 0.7, dark_overlay, 0.3, 0)
                        st.image(blended, caption="🔴 Detected Dark Regions", use_column_width=True)
                    
                    with col_b:
                        st.metric("Dark Regions", f"{suggestions['dark_regions']:.1f}%")
                        st.metric("Suggested Level", suggestions['level'])
                        st.metric("Suggested %", f"{suggestions['percentage']}%")
                else:
                    st.error("❌ Analysis failed")
        
        # REST OF EXISTING ANNOTATION CODE CONTINUES HERE...
def train_model_page():
    st.header("🤖 Train Deep Learning Model")
    
    # Check for annotated data
    annotation_files = [f for f in os.listdir("data/annotations") if f.endswith('.json')] if os.path.exists("data/annotations") else []
    
    if len(annotation_files) < 5:
        st.warning(f"Need at least 5 annotated images for training. Currently have: {len(annotation_files)}")
        st.info("💡 Go to 'Upload & Annotate' to create more training data")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        # Model naming section
        st.markdown("**Model Identity**")
        model_name = st.text_input(
            "Model name:",
            value=f"mycorrhizal_model_{datetime.now().strftime('%Y%m%d')}",
            help="Give your model a descriptive name for future use"
        )
        
        model_description = st.text_area(
            "Model description (optional):",
            placeholder="e.g., Trained on 50 root images from greenhouse experiment A",
            height=100
        )
        
        st.markdown("---")
        st.markdown("**Training Parameters**")
        
        epochs = st.slider("Number of epochs:", 1, 100, 20)
        learning_rate = st.selectbox("Learning rate:", [0.001, 0.0001, 0.00001], index=1)
        batch_size = st.selectbox("Batch size:", [8, 16, 32], index=1)
        
        # Model architecture selection
        model_type = st.selectbox(
            "Model architecture:",
            ["ResNet18", "ResNet34", "EfficientNetB0"]
        )
        
        use_gpu = st.checkbox("Use GPU if available", value=torch.cuda.is_available())
        
        # Data augmentation options
        st.subheader("Data Augmentation")
        augmentation = st.checkbox("Enable data augmentation", value=True)
        
        if augmentation:
            rotation_angle = st.slider("Max rotation angle:", 0, 45, 30)
            brightness_factor = st.slider("Brightness variation:", 0.0, 0.5, 0.2)
    
    with col2:
        st.subheader("Training Progress")
        
        # Show existing models
        existing_models = [f for f in os.listdir("models") if f.endswith('.pth')] if os.path.exists("models") else []
        if existing_models:
            st.markdown("**Existing Models:**")
            for model_file in existing_models:
                # Extract info from filename or metadata
                st.write(f"📁 {model_file}")
        
        if st.button("🚀 Start Training", type="primary"):
            # Validate model name
            if not model_name.strip():
                st.error("❌ Please provide a model name")
                return
            
            # Clean model name for filename
            safe_model_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_model_name = safe_model_name.replace(' ', '_')
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            
            try:
                # Initialize trainer
                trainer = ModelTrainer(
                    model_type=model_type,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    use_gpu=use_gpu
                )
                
                # Prepare data
                status_text.text("📊 Preparing training data...")
                trainer.prepare_data("data/raw", "data/annotations")
                
                # Training loop
                status_text.text("🤖 Training AI model...")
                for epoch in range(epochs):
                    loss = trainer.train_epoch()
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
                    
                    # Update metrics display
                    if epoch % 5 == 0:
                        metrics = trainer.get_training_metrics()
                        if metrics and len(metrics['train_loss']) > 0:
                            metrics_df = pd.DataFrame(metrics)
                            fig = px.line(metrics_df, x='epoch', y=['train_loss', 'val_loss'], 
                                        title="Training Progress")
                            metrics_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Save model with custom name and metadata
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_filename = f"{safe_model_name}_{timestamp}.pth"
                model_path = os.path.join("models", model_filename)
                
                # Create model metadata
                model_metadata = {
                    'model_name': model_name,
                    'description': model_description,
                    'model_type': model_type,
                    'training_date': datetime.now().isoformat(),
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'num_annotations': len(annotation_files),
                    'final_loss': trainer.train_losses[-1] if trainer.train_losses else None,
                    'final_accuracy': trainer.train_accuracies[-1] if trainer.train_accuracies else None
                }
                
                # Save model with metadata
                torch.save({
                    'model_state_dict': trainer.model.state_dict(),
                    'model_type': model_type,
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'train_losses': trainer.train_losses,
                    'val_losses': trainer.val_losses,
                    'train_accuracies': trainer.train_accuracies,
                    'val_accuracies': trainer.val_accuracies,
                    'metadata': model_metadata  # Add metadata
                }, model_path)
                
                # Save metadata separately for easy reading
                metadata_path = os.path.join("models", f"{safe_model_name}_{timestamp}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f, indent=2)
                
                st.success(f"🎉 Model '{model_name}' trained successfully!")
                st.success(f"📁 Saved as: {model_filename}")
                st.session_state.model_trained = True
                
                # Display model info
                st.subheader("📋 Model Information")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Name:** {model_name}")
                    st.write(f"**Architecture:** {model_type}")
                    st.write(f"**Training Images:** {len(annotation_files)}")
                with col_b:
                    st.write(f"**Final Loss:** {trainer.train_losses[-1]:.4f}")
                    st.write(f"**Final Accuracy:** {trainer.train_accuracies[-1]:.1f}%")
                    st.write(f"**File:** {model_filename}")
                
                # Final metrics display
                final_metrics = trainer.get_training_metrics()
                if final_metrics:
                    metrics_df = pd.DataFrame(final_metrics)
                    fig = px.line(metrics_df, x='epoch', y=['train_loss', 'val_loss'], 
                                title=f"Training Progress - {model_name}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model evaluation
                eval_results = trainer.evaluate_model()
                if eval_results:
                    st.subheader("📊 Model Evaluation")
                    st.write(f"**Accuracy:** {eval_results['classification_report']['accuracy']:.3f}")
                    
                    # Confusion matrix
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(eval_results['confusion_matrix'], 
                               annot=True, fmt='d', cmap='Blues',
                               xticklabels=eval_results['class_names'],
                               yticklabels=eval_results['class_names'], ax=ax)
                    ax.set_title(f'Confusion Matrix - {model_name}')
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.info("💡 Try reducing batch size or using a simpler model")
def batch_analysis_page():
    st.header("⚡ Batch AI Analysis")
    
    # Check for trained model
    model_files = [f for f in os.listdir("models") if f.endswith('.pth')] if os.path.exists("models") else []
    
    if not model_files:
        st.warning("❌ No trained models found. Please train a model first.")
        st.info("💡 Go to 'Train AI Model' to create your first model")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("AI Model Selection")
        
        # Enhanced model selection with metadata
        model_options = {}
        for model_file in model_files:
            try:
                # Try to load metadata
                metadata_file = model_file.replace('.pth', '_metadata.json')
                metadata_path = os.path.join("models", metadata_file)
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    display_name = f"{metadata.get('model_name', model_file)} ({metadata.get('training_date', '')[:10]})"
                else:
                    # Fallback to loading from model checkpoint
                    model_path = os.path.join("models", model_file)
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        metadata = checkpoint.get('metadata', {})
                        if metadata:
                            display_name = f"{metadata.get('model_name', model_file)} ({metadata.get('training_date', '')[:10]})"
                        else:
                            display_name = model_file
                    except:
                        display_name = model_file
                
                model_options[display_name] = model_file
            except:
                model_options[model_file] = model_file
        
        selected_display_name = st.selectbox("Choose trained model:", list(model_options.keys()))
        selected_model = model_options[selected_display_name]
        
        # Enhanced model info display
        if selected_model:
            model_path = os.path.join("models", selected_model)
            try:
                # Try to load metadata first
                metadata_file = selected_model.replace('.pth', '_metadata.json')
                metadata_path = os.path.join("models", metadata_file)
                
                model_info = {}
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    model_info = {
                        "Model Name": metadata.get('model_name', 'Unknown'),
                        "Description": metadata.get('description', 'No description'),
                        "Architecture": metadata.get('model_type', 'Unknown'),
                        "Training Date": metadata.get('training_date', '')[:10],
                        "Epochs": metadata.get('epochs', 'Unknown'),
                        "Training Images": metadata.get('num_annotations', 'Unknown'),
                        "Final Accuracy": f"{metadata.get('final_accuracy', 0):.1f}%" if metadata.get('final_accuracy') else 'Unknown'
                    }
                else:
                    # Fallback to checkpoint data
                    checkpoint = torch.load(model_path, map_location='cpu')
                    metadata = checkpoint.get('metadata', {})
                    model_info = {
                        "Model Type": checkpoint.get('model_type', 'Unknown'),
                        "File Size": f"{os.path.getsize(model_path) / (1024*1024):.1f} MB"
                    }
                    if metadata:
                        model_info.update({
                            "Model Name": metadata.get('model_name', 'Unknown'),
                            "Training Date": metadata.get('training_date', '')[:10],
                        })
                
                # Display model information
                st.markdown("**Model Information:**")
                for key, value in model_info.items():
                    if value and value != 'Unknown':
                        st.write(f"**{key}:** {value}")
                        
            except Exception as e:
                st.warning(f"Could not load model info: {e}")
        
        # Rest of the batch analysis code remains the same...
        st.subheader("Batch Upload")
        batch_files = st.file_uploader(
            "Upload images for AI analysis",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif']
        )
        
        # Analysis parameters
        confidence_threshold = st.slider("Confidence threshold:", 0.0, 1.0, 0.7)
        grid_size = st.selectbox("Grid size for quantification:", [10, 20, 50, 100], index=2)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            save_visualizations = st.checkbox("Save prediction visualizations", value=True)
            export_gradcam = st.checkbox("Generate Grad-CAM heatmaps", value=False)
            detailed_features = st.checkbox("Detailed feature analysis", value=True)
    
    # Rest of the function continues as before...
    with col2:
        st.subheader("AI Analysis Results")
        
        if batch_files and st.button("🤖 Analyze with AI", type="primary"):
            # Initialize AI components
            model_path = os.path.join("models", selected_model)
            inference_engine = ModelInference(model_path)
            quantifier = ColonizationQuantifier(grid_size=grid_size)
            
            if export_gradcam:
                gradcam_viz = GradCAMVisualizer(model_path)
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(batch_files):
                status_text.text(f"🔍 Analyzing {uploaded_file.name}...")
                
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Run AI inference
                    prediction = inference_engine.predict(temp_path, confidence_threshold)
                    
                    # AI-powered quantification
                    quantification = quantifier.quantify_colonization(temp_path, prediction)
                    
                    # Grad-CAM analysis (if requested)
                    gradcam_path = None
                    if export_gradcam:
                        try:
                            gradcam_image = gradcam_viz.generate_gradcam(temp_path)
                            gradcam_path = f"data/results/gradcam_{uploaded_file.name}"
                            cv2.imwrite(gradcam_path, gradcam_image)
                        except Exception as e:
                            st.warning(f"Grad-CAM failed for {uploaded_file.name}: {e}")
                    
                    result = {
                        "filename": uploaded_file.name,
                        "model_used": selected_display_name,
                        "ai_predicted_class": prediction["class_name"],
                        "ai_confidence": prediction["confidence"],
                        "colonization_percentage": quantification["colonization_percentage"],
                        "detected_features": prediction["detected_features"],
                        "grid_method_percentage": quantification["grid_method"]["percentage"],
                        "area_method_percentage": quantification["area_method"]["percentage"],
                        "intensity_method_percentage": quantification["intensity_method"]["percentage"],
                        "consensus_confidence": quantification["consensus_confidence"],
                        "above_threshold": prediction["above_threshold"],
                        "gradcam_available": gradcam_path is not None,
                        "analysis_timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing {uploaded_file.name}: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                progress_bar.progress((i + 1) / len(batch_files))
            
            if results:
                # Save results with model info
                results_df = pd.DataFrame(results)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_file = f"data/results/ai_batch_analysis_{timestamp}.csv"
                results_df.to_csv(results_file, index=False)
                
                # Display results
                st.subheader("📊 AI Analysis Summary")
                
                # Key metrics
                avg_colonization = results_df["colonization_percentage"].mean()
                high_confidence_count = len(results_df[results_df["above_threshold"]])
                total_count = len(results_df)
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Average Colonization", f"{avg_colonization:.1f}%")
                metric_col2.metric("High Confidence", f"{high_confidence_count}/{total_count}")
                metric_col3.metric("Model Used", selected_display_name.split(' (')[0])
                
                # Results table
                st.dataframe(results_df)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution of colonization
                    fig = px.histogram(results_df, x="colonization_percentage", 
                                     title="AI-Detected Colonization Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Confidence vs Colonization
                    fig = px.scatter(results_df, x="ai_confidence", y="colonization_percentage",
                                   color="ai_predicted_class",
                                   title="AI Confidence vs Colonization")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ AI analysis complete! Results saved to: {results_file}")
        else:
            st.info("👆 Upload images and click 'Analyze with AI' to start")

def results_export_page():
    st.header("📊 Results & Export")
    
    # List available results
    result_files = [f for f in os.listdir("data/results") if f.endswith('.csv')] if os.path.exists("data/results") else []
    
    if not result_files:
        st.warning("❌ No analysis results found.")
        st.info("💡 Run AI analysis in 'Batch Analysis' to generate results")
        return
    
    tab1, tab2, tab3 = st.tabs(["📈 Analysis Results", "📊 Visualizations", "📤 Export"])
    
    with tab1:
        st.subheader("Available Results")
        selected_result = st.selectbox("Choose result file:", result_files)
        
        if selected_result:
            result_path = os.path.join("data/results", selected_result)
            df = pd.read_csv(result_path)
            
            # Display data
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            if "colonization_percentage" in df.columns:
                stats = df["colonization_percentage"].describe()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean", f"{stats['mean']:.1f}%")
                col2.metric("Std Dev", f"{stats['std']:.1f}%")
                col3.metric("Min", f"{stats['min']:.1f}%")
                col4.metric("Max", f"{stats['max']:.1f}%")
            
            # AI Performance metrics
            if "ai_confidence" in df.columns:
                st.subheader("🤖 AI Performance")
                avg_confidence = df["ai_confidence"].mean()
                high_conf_ratio = len(df[df["above_threshold"]]) / len(df)
                
                col1, col2 = st.columns(2)
                col1.metric("Average AI Confidence", f"{avg_confidence:.3f}")
                col2.metric("High Confidence Rate", f"{high_conf_ratio:.1%}")
    
    with tab2:
        st.subheader("Data Visualizations")
        
        if selected_result and 'df' in locals():
            viz_type = st.selectbox(
                "Choose visualization:",
                ["Colonization Distribution", "AI Confidence Analysis", "Method Comparison", "Feature Detection"]
            )
            
            if viz_type == "Colonization Distribution":
                fig = px.histogram(df, x="colonization_percentage", 
                                 title="Distribution of Colonization Percentages",
                                 nbins=20)
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "AI Confidence Analysis":
                if "ai_confidence" in df.columns:
                    fig = px.scatter(df, x="ai_confidence", y="colonization_percentage",
                                   color="ai_predicted_class",
                                   title="AI Confidence vs Colonization",
                                   hover_data=["filename"])
                    st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Method Comparison":
                methods = ["grid_method_percentage", "area_method_percentage", "intensity_method_percentage"]
                available_methods = [m for m in methods if m in df.columns]
                
                if available_methods:
                    method_data = df[available_methods + ["colonization_percentage"]].melt(
                        id_vars=["colonization_percentage"],
                        var_name="method",
                        value_name="percentage"
                    )
                    
                    fig = px.scatter(method_data, x="percentage", y="colonization_percentage",
                                   color="method",
                                   title="Quantification Method Comparison")
                    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, 
                                line=dict(dash="dash", color="gray"))
                    st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Feature Detection":
                if "detected_features" in df.columns:
                    # Parse detected features
                    all_features = []
                    for features_str in df["detected_features"]:
                        if isinstance(features_str, str) and features_str:
                            features = features_str.split(", ") if ", " in features_str else [features_str]
                            all_features.extend(features)
                    
                    if all_features:
                        feature_counts = pd.Series(all_features).value_counts()
                        fig = px.bar(x=feature_counts.index, y=feature_counts.values,
                                   title="Detected Mycorrhizal Features")
                        fig.update_xaxis(title="Feature Type")
                        fig.update_yaxis(title="Detection Count")
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Export Options")
        
        if selected_result and 'df' in locals():
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Standard Exports**")
                
                # CSV download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"mycorrhizal_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )
                
                # JSON export
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"mycorrhizal_results_{datetime.now().strftime('%Y%m%d')}.json",
                    mime='application/json'
                )
            
            with col2:
                st.write("**Research Exports**")
                
                # Summary report
                if st.button("📄 Generate Summary Report"):
                    report = generate_summary_report(df)
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"mycorrhizal_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime='text/plain'
                    )
                
                # Statistical analysis
                if st.button("📊 Generate Statistics"):
                    stats = generate_statistical_analysis(df)
                    st.download_button(
                        label="📥 Download Statistics",
                        data=stats,
                        file_name=f"mycorrhizal_stats_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime='text/plain'
                    )

def explainability_page():
    st.header("🔍 AI Model Explainability (Grad-CAM)")
    
    # Check for trained model
    model_files = [f for f in os.listdir("models") if f.endswith('.pth')] if os.path.exists("models") else []
    
    if not model_files:
        st.warning("❌ No trained models found.")
        st.info("💡 Train a model first to use explainability features")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Upload Image for AI Explanation")
        
        selected_model = st.selectbox("Choose AI model:", model_files)
        
        uploaded_file = st.file_uploader(
            "Upload image for Grad-CAM analysis",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif']
        )
        
        if uploaded_file:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Show AI prediction
            if st.button("🤖 Get AI Prediction"):
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    model_path = os.path.join("models", selected_model)
                    inference_engine = ModelInference(model_path)
                    prediction = inference_engine.predict(temp_path)
                    
                    st.success("✅ AI Analysis Complete!")
                    st.write(f"**Predicted Class:** {prediction['class_name']}")
                    st.write(f"**Confidence:** {prediction['confidence']:.3f}")
                    st.write(f"**Detected Features:** {', '.join(prediction['detected_features'])}")
                    
                    # Store prediction in session state
                    st.session_state.current_prediction = prediction
                    st.session_state.current_temp_path = temp_path
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    with col2:
        st.subheader("🔥 Grad-CAM Heatmap Visualization")
        
        if uploaded_file and st.button("🔍 Generate Grad-CAM", type="primary"):
            if 'current_temp_path' in st.session_state:
                try:
                    # Initialize Grad-CAM visualizer
                    model_path = os.path.join("models", selected_model)
                    gradcam_viz = GradCAMVisualizer(model_path)
                    
                    # Generate Grad-CAM visualization
                    with st.spinner("🔄 Generating AI explanation..."):
                        gradcam_image = gradcam_viz.generate_gradcam(st.session_state.current_temp_path)
                        # Get raw gradcam for attention analysis
                        raw_gradcam = gradcam_viz.generate_gradcam(st.session_state.current_temp_path, return_raw=True)
                    
                    # Display Grad-CAM
                    st.image(gradcam_image, caption="Grad-CAM: What AI Sees", use_column_width=True)
                    
                    st.success("✅ Grad-CAM visualization generated!")
                    
                    # Attention analysis using raw gradcam
                    try:
                        attention_analysis = gradcam_viz.analyze_attention_regions(
                            st.session_state.current_temp_path, 
                            raw_gradcam  # Use raw gradcam, not the visualization
                        )
                        
                        st.subheader("📊 Attention Analysis")
                        st.write(f"**High Attention Regions:** {attention_analysis['num_attention_regions']}")
                        st.write(f"**Total Attention Area:** {attention_analysis['attention_percentage']:.1f}%")
                        
                        if attention_analysis['top_regions']:
                            st.write("**Top Attention Regions:**")
                            for i, region in enumerate(attention_analysis['top_regions'][:3]):
                                st.write(f"Region {i+1}: Center({region['center'][0]}, {region['center'][1]}), "
                                       f"Attention: {region['avg_attention']:.3f}")
                    except Exception as e:
                        st.warning(f"⚠️ Attention analysis failed: {e}")
                    
                    # Clean up
                    if os.path.exists(st.session_state.current_temp_path):
                        os.remove(st.session_state.current_temp_path)
                        
                except Exception as e:
                    st.error(f"❌ Grad-CAM generation failed: {str(e)}")
                    st.info("💡 This might be due to model compatibility issues")
        
        # Class-specific Grad-CAM
        if uploaded_file:
            st.subheader("🎯 Class-Specific Analysis")
            
            target_class = st.selectbox(
                "Generate Grad-CAM for specific class:",
                ["Predicted Class", "Not colonized", "Lightly colonized", 
                 "Moderately colonized", "Heavily colonized"]
            )
            
            if st.button("🔍 Generate Class-Specific Grad-CAM"):
                if 'current_temp_path' in st.session_state:
                    try:
                        model_path = os.path.join("models", selected_model)
                        gradcam_viz = GradCAMVisualizer(model_path)
                        
                        # Map class names to indices
                        class_mapping = {
                            "Not colonized": 0,
                            "Lightly colonized": 1,
                            "Moderately colonized": 2,
                            "Heavily colonized": 3
                        }
                        
                        target_idx = None if target_class == "Predicted Class" else class_mapping.get(target_class)
                        
                        with st.spinner(f"🔄 Generating Grad-CAM for {target_class}..."):
                            gradcam_image = gradcam_viz.generate_gradcam(
                                st.session_state.current_temp_path, 
                                target_class=target_idx
                            )
                        
                        st.image(gradcam_image, caption=f"Grad-CAM for: {target_class}", 
                               use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Class-specific Grad-CAM failed: {e}")
        
        # Additional information
        if uploaded_file:
            st.markdown("---")
            st.markdown("**💡 Understanding Grad-CAM:**")
            st.markdown("- **Red/Yellow areas**: High attention regions where the AI focuses")
            st.markdown("- **Blue areas**: Low attention regions")
            st.markdown("- **Attention analysis**: Shows which parts of the image influenced the AI's decision")
            st.markdown("- **Class-specific**: See what the AI would focus on for different colonization levels")

# ADD this new page function to app.py:
def smart_annotation_page():
    st.header("🧠 Smart Annotation Assistant")
    st.markdown("### AI suggests which images to annotate next for maximum learning efficiency")
    
    # Check current status
    all_images = []
    if os.path.exists("data/raw"):
        all_images = [f for f in os.listdir("data/raw") 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    
    annotated_images = []
    if os.path.exists("data/annotations"):
        annotation_files = [f for f in os.listdir("data/annotations") if f.endswith('.json')]
        for ann_file in annotation_files:
            try:
                with open(os.path.join("data/annotations", ann_file), 'r') as f:
                    data = json.load(f)
                    if 'image' in data:
                        annotated_images.append(data['image'])
            except:
                continue
    
    unlabeled_images = [img for img in all_images if img not in annotated_images]
    
    # Status display
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", len(all_images))
    col2.metric("Annotated", len(annotated_images))
    col3.metric("Remaining", len(unlabeled_images))
    
    if len(annotated_images) >= 3:
        progress = len(annotated_images) / len(all_images) if all_images else 0
        st.progress(progress)
        st.write(f"📊 **Progress:** {progress:.1%} complete")
    
    if len(unlabeled_images) == 0:
        st.success("🎉 All images annotated!")
        return
    
    # Smart selection
    st.subheader("🎯 Smart Image Selection")
    
    n_select = st.slider("Number of images to select:", 1, min(10, len(unlabeled_images)), 5)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button("🤖 Find Most Important Images", type="primary"):
            with st.spinner("🔍 Analyzing images for optimal selection..."):
                
                # Initialize active learning
                latest_model = None
                if os.path.exists("models"):
                    model_files = [f for f in os.listdir("models") if f.endswith('.pth')]
                    if model_files:
                        latest_model = os.path.join("models", model_files[-1])
                
                selector = ActiveLearningSelector(latest_model)
                
                # Get full paths
                unlabeled_paths = [os.path.join("data/raw", img) for img in unlabeled_images]
                annotated_paths = [os.path.join("data/raw", img) for img in annotated_images]
                
                # Select important images
                selected_images = selector.select_next_images(
                    unlabeled_paths, annotated_paths, n_select
                )
                
                if selected_images:
                    st.session_state.selected_for_annotation = selected_images
                    st.success(f"🎯 Selected {len(selected_images)} high-priority images!")
                else:
                    st.error("❌ Could not analyze images")
    
    with col_b:
        # Show selection strategy info
        if len(annotated_images) < 10:
            st.info("🔄 **Strategy:** Diversity Sampling")
            st.write("Selecting diverse images to cover different visual patterns")
        else:
            st.info("🧠 **Strategy:** Uncertainty + Diversity")
            st.write("AI focuses on images where it's most uncertain + diverse samples")
    
    # Display selected images for annotation
    if 'selected_for_annotation' in st.session_state:
        st.subheader("📝 Priority Images for Annotation")
        
        selected_images = st.session_state.selected_for_annotation
        
        for i, img_info in enumerate(selected_images):
            st.markdown("---")
            st.subheader(f"🎯 Priority {i+1}: {os.path.basename(img_info['image_path'])}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                try:
                    image = Image.open(img_info['image_path'])
                    st.image(image, use_column_width=True)
                    
                    # Show AI insights
                    if 'uncertainty' in img_info:
                        st.metric("🎲 AI Uncertainty", f"{img_info['uncertainty']:.2f}")
                    if 'confidence' in img_info:
                        st.metric("🎯 AI Confidence", f"{img_info['confidence']:.1%}")
                    if 'predicted_class' in img_info:
                        st.info(f"🤖 **AI Guess:** {img_info['predicted_class']}")
                    if 'reason' in img_info:
                        st.info(f"📋 **Why Selected:** {img_info['reason']}")
                        
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    continue
            
            with col2:
                st.markdown("**Quick Annotation:**")
                
                # Quick annotation interface
                colonization = st.selectbox(
                    "Colonization level:",
                    ["Not colonized", "Lightly colonized", "Moderately colonized", "Heavily colonized"],
                    key=f"quick_col_{i}"
                )
                
                percentage = st.slider(
                    "Percentage:", 0, 100, 25, key=f"quick_pct_{i}"
                )
                
                features = st.multiselect(
                    "Detected features:",
                    ["Arbuscules", "Vesicles", "Hyphae", "Spores", "Entry points"],
                    key=f"quick_feat_{i}"
                )
                
                notes = st.text_area(
                    "Notes (optional):", 
                    placeholder="Any observations...",
                    key=f"quick_notes_{i}",
                    height=80
                )
                
                if st.button(f"💾 Save Priority {i+1}", key=f"save_quick_{i}", type="primary"):
                    # Save annotation
                    img_name = os.path.basename(img_info['image_path'])
                    annotation_file = os.path.join("data/annotations", f"{img_name}_annotation.json")
                    
                    annotation_data = {
                        "image": img_name,
                        "annotation_type": colonization,
                        "colonization_percentage": percentage,
                        "detected_features": features,
                        "notes": notes,
                        "timestamp": datetime.now().isoformat(),
                        "annotation_method": "smart_selection",
                        "ai_suggestion": img_info.get('predicted_class', None),
                        "ai_confidence": img_info.get('confidence', None)
                    }
                    
                    try:
                        with open(annotation_file, 'w') as f:
                            json.dump(annotation_data, f, indent=2)
                        
                        st.success(f"✅ Saved annotation for Priority {i+1}!")
                        
                        # Remove from session state
                        st.session_state.selected_for_annotation.remove(img_info)
                        
                        # Update metrics
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error saving: {e}")
    
    # Efficiency estimator
    st.sidebar.header("💡 Efficiency Calculator")
    total_to_annotate = st.sidebar.number_input("Total images to process:", 10, 5000, 100)
    manual_rate = st.sidebar.number_input("Your annotation rate (per hour):", 1, 50, 15)
    
    # Traditional vs Smart annotation time
    traditional_time = total_to_annotate / manual_rate
    smart_needed = int(total_to_annotate * 0.3)  # Active learning typically needs 30%
    smart_time = smart_needed / manual_rate
    
    st.sidebar.metric("📚 Traditional Method", f"{traditional_time:.1f} hours")
    st.sidebar.metric("🧠 Smart Method", f"{smart_time:.1f} hours")
    st.sidebar.metric("⏰ Time Saved", f"{traditional_time - smart_time:.1f} hours")
    st.sidebar.metric("📊 Efficiency Gain", f"{(traditional_time - smart_time)/traditional_time:.1%}")

page = st.sidebar.selectbox(
    "Choose a page:",
    ["Upload & Annotate", "Smart Annotation", "Train AI Model", "Batch Analysis", "Results & Export", "Model Explainability"]
)

# And add the new page to the if/elif chain:
elif page == "Smart Annotation":
    smart_annotation_page()
def generate_summary_report(df):
    """Generate a text summary report"""
    report = f"""
MYCORRHIZAL COLONIZATION ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET SUMMARY
===============
Total samples analyzed: {len(df)}
Average colonization: {df['colonization_percentage'].mean():.2f}%
Standard deviation: {df['colonization_percentage'].std():.2f}%
Range: {df['colonization_percentage'].min():.1f}% - {df['colonization_percentage'].max():.1f}%

AI PERFORMANCE
==============
"""
    
    if 'ai_confidence' in df.columns:
        avg_confidence = df['ai_confidence'].mean()
        high_conf_count = len(df[df['above_threshold']])
        report += f"""Average AI confidence: {avg_confidence:.3f}
High confidence predictions: {high_conf_count}/{len(df)} ({high_conf_count/len(df)*100:.1f}%)
"""
    
    if 'ai_predicted_class' in df.columns:
        class_distribution = df['ai_predicted_class'].value_counts()
        report += f"""
CLASS DISTRIBUTION
==================
"""
        for class_name, count in class_distribution.items():
            percentage = count / len(df) * 100
            report += f"{class_name}: {count} samples ({percentage:.1f}%)\n"
    
    return report

def generate_statistical_analysis(df):
    """Generate statistical analysis"""
    stats = f"""
STATISTICAL ANALYSIS
===================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DESCRIPTIVE STATISTICS
=====================
"""
    
    if 'colonization_percentage' in df.columns:
        desc_stats = df['colonization_percentage'].describe()
        for stat, value in desc_stats.items():
            stats += f"{stat}: {value:.3f}\n"
    
    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        stats += f"""
CORRELATION MATRIX
==================
{corr_matrix.to_string()}
"""
    
    return stats

if __name__ == "__main__":
    main()
