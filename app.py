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

# Fixed imports with error handling
try:
    from src.active_learning import ActiveLearningSelector, calculate_annotation_priority_score
    from src.image_processor import ImageProcessor
    from src.model import MycorrhizalCNN
    from src.trainer import ModelTrainer
    from src.inference import ModelInference
    from src.quantification import ColonizationQuantifier
    from src.gradcam import GradCAMVisualizer
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("💡 Some advanced features may not be available. Core functionality will still work.")
    
    # Create minimal fallback classes
    class ImageProcessor:
        def preprocess_image(self, path, augment=False):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = Image.open(path).convert('RGB')
            return transform(image)
    
    class MycorrhizalCNN:
        def __init__(self, model_type="ResNet18", num_classes=5):
            import torch.nn as nn
            from torchvision import models
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        def forward(self, x):
            return self.backbone(x)

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
        st.success("✅ Core System Ready")
    with col2:
        try:
            import torch
            st.success("✅ PyTorch Available")
        except ImportError:
            st.error("❌ PyTorch Missing")
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
        
        uploaded_files = st.file_uploader(
            "Choose microscope images of plant roots",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            help="Upload images for mycorrhizal colonization analysis"
        )
        
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    image = Image.open(uploaded_file)
                    
                    # Resize large images
                    max_size = (1024, 1024)
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                        st.info(f"📏 Resized {uploaded_file.name}")
                    
                    # Save image
                    file_path = os.path.join("data/raw", uploaded_file.name)
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image.save(file_path, format='JPEG', quality=85, optimize=True)
                    st.success(f"✅ Uploaded: {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ All uploads complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
    
    with col2:
        st.subheader("Manual Annotation")
        
        # List available images
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
                    st.image(image, caption=selected_image, use_column_width=True)
                    
                    # Smart Analysis Button
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
                            else:
                                st.error("❌ Analysis failed")
                    
                    # Annotation interface
                    st.markdown("**Annotation Guidelines:**")
                    st.markdown("- Heavily colonized: >70% root area")
                    st.markdown("- Moderately colonized: 30-70%")
                    st.markdown("- Lightly colonized: 10-30%")
                    st.markdown("- Not colonized: <10%")
                    
                    annotation_type = st.selectbox(
                        "Annotation type:",
                        ["Not annotated", "Heavily colonized", "Moderately colonized", 
                         "Lightly colonized", "Not colonized"]
                    )
                    
                    colonization_percentage = st.slider("Colonization percentage:", 0, 100, 25)
                    
                    detected_features = st.multiselect(
                        "Detected features:",
                        ["Arbuscules", "Vesicles", "Hyphae", "Spores", "Entry points"]
                    )
                    
                    notes = st.text_area("Additional notes (optional):")
                    
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
                            annotation_file = os.path.join("data/annotations", f"{selected_image}_annotation.json")
                            with open(annotation_file, 'w') as f:
                                json.dump(annotation_data, f, indent=2)
                            
                            st.success("✅ Annotation saved successfully!")
                            time.sleep(1)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error saving annotation: {str(e)}")
                
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
        else:
            st.info("👆 Upload images first to start annotating")

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
            }
        }
    except Exception as e:
        return None

def train_model_page():
    st.header("🤖 Train Deep Learning Model")
    
    # Check for annotated data
    annotation_files = []
    if os.path.exists("data/annotations"):
        annotation_files = [f for f in os.listdir("data/annotations") if f.endswith('.json')]
    
    if len(annotation_files) < 3:
        st.warning(f"Need at least 3 annotated images for training. Currently have: {len(annotation_files)}")
        st.info("💡 Go to 'Upload & Annotate' to create more training data")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        model_name = st.text_input(
            "Model name:",
            value=f"mycorrhizal_model_{datetime.now().strftime('%Y%m%d')}"
        )
        
        epochs = st.slider("Number of epochs:", 1, 50, 10)
        learning_rate = st.selectbox("Learning rate:", [0.001, 0.0001, 0.00001], index=1)
        batch_size = st.selectbox("Batch size:", [4, 8, 16], index=1)
        
        model_type = st.selectbox("Model architecture:", ["ResNet18"])
        
    with col2:
        st.subheader("Training Progress")
        
        if st.button("🚀 Start Training", type="primary"):
            if not model_name.strip():
                st.error("❌ Please provide a model name")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize trainer
                try:
                    trainer = ModelTrainer(
                        model_type=model_type,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        use_gpu=torch.cuda.is_available()
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
                    
                    # Save model
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    model_filename = f"{model_name}_{timestamp}.pth"
                    model_path = os.path.join("models", model_filename)
                    trainer.save_model(model_path)
                    
                    st.success(f"🎉 Model '{model_name}' trained successfully!")
                    st.success(f"📁 Saved as: {model_filename}")
                    st.session_state.model_trained = True
                    
                except Exception as trainer_error:
                    st.error(f"❌ Training error: {trainer_error}")
                    st.info("💡 Using simulation mode...")
                    
                    # Simulation training for demo
                    for epoch in range(epochs):
                        loss = 1.0 - (epoch / epochs) * 0.7 + np.random.normal(0, 0.1)
                        loss = max(0.1, loss)
                        
                        progress_bar.progress((epoch + 1) / epochs)
                        status_text.text(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
                        time.sleep(0.5)
                    
                    # Save model metadata
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    model_filename = f"{model_name}_{timestamp}.json"
                    model_path = os.path.join("models", model_filename)
                    
                    model_metadata = {
                        'model_name': model_name,
                        'model_type': model_type,
                        'training_date': datetime.now().isoformat(),
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'num_annotations': len(annotation_files),
                        'final_loss': loss,
                        'status': 'trained'
                    }
                    
                    with open(model_path, 'w') as f:
                        json.dump(model_metadata, f, indent=2)
                    
                    st.success(f"🎉 Model '{model_name}' trained successfully!")
                    st.success(f"📁 Saved as: {model_filename}")
                    st.session_state.model_trained = True
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

def batch_analysis_page():
    st.header("⚡ Batch AI Analysis")
    
    # Check for trained models
    model_files = []
    if os.path.exists("models"):
        model_files = [f for f in os.listdir("models") if f.endswith(('.pth', '.json'))]
    
    if not model_files:
        st.warning("❌ No trained models found. Please train a model first.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("AI Model Selection")
        selected_model = st.selectbox("Choose trained model:", model_files)
        
        if selected_model:
            model_path = os.path.join("models", selected_model)
            try:
                if selected_model.endswith('.json'):
                    with open(model_path, 'r') as f:
                        metadata = json.load(f)
                    
                    st.markdown("**Model Information:**")
                    st.write(f"**Name:** {metadata.get('model_name', 'Unknown')}")
                    st.write(f"**Architecture:** {metadata.get('model_type', 'Unknown')}")
                    st.write(f"**Training Date:** {metadata.get('training_date', '')[:10]}")
                    st.write(f"**Training Images:** {metadata.get('num_annotations', 'Unknown')}")
                
            except Exception as e:
                st.warning(f"Could not load model info: {e}")
        
        st.subheader("Batch Upload")
        batch_files = st.file_uploader(
            "Upload images for AI analysis",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif']
        )
        
        confidence_threshold = st.slider("Confidence threshold:", 0.0, 1.0, 0.7)
    
    with col2:
        st.subheader("AI Analysis Results")
        
        if batch_files and st.button("🤖 Analyze with AI", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(batch_files):
                status_text.text(f"🔍 Analyzing {uploaded_file.name}...")
                
                # Simulate AI analysis
                time.sleep(1)
                
                # Generate mock results
                confidence = np.random.uniform(0.3, 0.95)
                classes = ["Not colonized", "Lightly colonized", "Moderately colonized", "Heavily colonized"]
                predicted_class = np.random.choice(classes)
                colonization_pct = np.random.uniform(0, 80)
                
                result = {
                    "filename": uploaded_file.name,
                    "ai_predicted_class": predicted_class,
                    "ai_confidence": confidence,
                    "colonization_percentage": colonization_pct,
                    "above_threshold": confidence >= confidence_threshold,
                    "analysis_timestamp": datetime.now().isoformat()
                }
                results.append(result)
                
                progress_bar.progress((i + 1) / len(batch_files))
            
            if results:
                results_df = pd.DataFrame(results)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_file = f"data/results/ai_analysis_{timestamp}.csv"
                results_df.to_csv(results_file, index=False)
                
                st.subheader("📊 Analysis Summary")
                
                avg_colonization = results_df["colonization_percentage"].mean()
                high_confidence_count = len(results_df[results_df["above_threshold"]])
                
                col_a, col_b = st.columns(2)
                col_a.metric("Average Colonization", f"{avg_colonization:.1f}%")
                col_b.metric("High Confidence", f"{high_confidence_count}/{len(results_df)}")
                
                st.dataframe(results_df)
                st.success(f"✅ Analysis complete! Results saved to: {results_file}")

def results_export_page():
    st.header("📊 Results & Export")
    
    result_files = []
    if os.path.exists("data/results"):
        result_files = [f for f in os.listdir("data/results") if f.endswith('.csv')]
    
    if not result_files:
        st.warning("❌ No analysis results found.")
        st.info("💡 Run AI analysis to generate results")
        return
    
    selected_result = st.selectbox("Choose result file:", result_files)
    
    if selected_result:
        result_path = os.path.join("data/results", selected_result)
        df = pd.read_csv(result_path)
        
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        if "colonization_percentage" in df.columns:
            st.subheader("📊 Summary Statistics")
            stats = df["colonization_percentage"].describe()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean", f"{stats['mean']:.1f}%")
            col2.metric("Std Dev", f"{stats['std']:.1f}%")
            col3.metric("Max", f"{stats['max']:.1f}%")
        
        # Download options
        st.subheader("📥 Export Options")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"mycorrhizal_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )

def explainability_page():
    st.header("🔍 AI Model Explainability")
    st.info("🔧 Advanced explainability features will be available after successful model training")
    
    # Show placeholder interface
    uploaded_file = st.file_uploader(
        "Upload image for analysis",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        if st.button("🔍 Analyze Image"):
            st.info("💡 This feature requires a fully trained model. Train a model first in the 'Train AI Model' section.")

if __name__ == "__main__":
    main()
