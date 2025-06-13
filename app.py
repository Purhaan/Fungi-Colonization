import streamlit as st
import os
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
        ["Upload & Annotate", "Train Model", "Batch Analysis", "Results & Export", "Model Explainability"]
    )
    
    if page == "Upload & Annotate":
        upload_and_annotate_page()
    elif page == "Train Model":
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
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif']
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save uploaded file
                file_path = os.path.join("data/raw", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded: {uploaded_file.name}")
    
    with col2:
        st.subheader("Manual Annotation")
        
        # List available images
        image_files = [f for f in os.listdir("data/raw") if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        
        if image_files:
            selected_image = st.selectbox("Select image to annotate:", image_files)
            
            if selected_image:
                image_path = os.path.join("data/raw", selected_image)
                image = Image.open(image_path)
                
                # Display image
                st.image(image, caption=selected_image, use_column_width=True)
                
                # Annotation interface
                st.markdown("**Annotation Guidelines:**")
                st.markdown("- Green regions: Colonized areas (arbuscules, vesicles, hyphae)")
                st.markdown("- Red regions: Non-colonized areas")
                
                # Simple annotation using selectbox regions
                annotation_type = st.selectbox(
                    "Annotation type for this image:",
                    ["Not annotated", "Heavily colonized", "Moderately colonized", "Lightly colonized", "Not colonized"]
                )
                
                colonization_percentage = st.slider(
                    "Estimated colonization percentage:",
                    0, 100, 50 if annotation_type != "Not annotated" else 0
                )
                
                if st.button("Save Annotation"):
                    annotation_data = {
                        "image": selected_image,
                        "annotation_type": annotation_type,
                        "colonization_percentage": colonization_percentage,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Save annotation
                    annotation_file = os.path.join("data/annotations", f"{selected_image}_annotation.json")
                    with open(annotation_file, 'w') as f:
                        json.dump(annotation_data, f, indent=2)
                    
                    st.success("Annotation saved!")
                    
                    # Update session state
                    st.session_state.annotation_data[selected_image] = annotation_data

def train_model_page():
    st.header("🤖 Train Deep Learning Model")
    
    # Check for annotated data
    annotation_files = [f for f in os.listdir("data/annotations") if f.endswith('.json')]
    
    if len(annotation_files) < 5:
        st.warning(f"Need at least 5 annotated images for training. Currently have: {len(annotation_files)}")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        epochs = st.slider("Number of epochs:", 1, 100, 20)
        learning_rate = st.selectbox("Learning rate:", [0.001, 0.0001, 0.00001], index=1)
        batch_size = st.selectbox("Batch size:", [8, 16, 32], index=1)
        
        # Model architecture selection
        model_type = st.selectbox(
            "Model architecture:",
            ["ResNet18", "ResNet34", "EfficientNetB0"]
        )
        
        use_gpu = st.checkbox("Use GPU if available", value=True)
        
    with col2:
        st.subheader("Training Progress")
        
        if st.button("Start Training", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize trainer
                trainer = ModelTrainer(
                    model_type=model_type,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    use_gpu=use_gpu
                )
                
                # Prepare data
                status_text.text("Preparing training data...")
                trainer.prepare_data("data/raw", "data/annotations")
                
                # Train model
                status_text.text("Training model...")
                for epoch in range(epochs):
                    loss = trainer.train_epoch()
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
                
                # Save model
                model_path = f"models/mycorrhizal_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                trainer.save_model(model_path)
                
                st.success(f"Model trained successfully! Saved to: {model_path}")
                st.session_state.model_trained = True
                
                # Display training metrics
                metrics_df = pd.DataFrame(trainer.get_training_metrics())
                if not metrics_df.empty:
                    fig = px.line(metrics_df, x='epoch', y=['train_loss', 'val_loss'], 
                                title="Training Progress")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

def batch_analysis_page():
    st.header("⚡ Batch Analysis")
    
    # Check for trained model
    model_files = [f for f in os.listdir("models") if f.endswith('.pth')]
    
    if not model_files:
        st.warning("No trained models found. Please train a model first.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Selection")
        selected_model = st.selectbox("Choose trained model:", model_files)
        
        st.subheader("Batch Upload")
        batch_files = st.file_uploader(
            "Upload images for batch analysis",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif']
        )
        
        # Analysis parameters
        confidence_threshold = st.slider("Confidence threshold:", 0.0, 1.0, 0.7)
        grid_size = st.selectbox("Grid size for quantification:", [10, 20, 50, 100], index=2)
        
    with col2:
        st.subheader("Analysis Results")
        
        if batch_files and st.button("Analyze Batch", type="primary"):
            # Initialize inference engine
            model_path = os.path.join("models", selected_model)
            inference_engine = ModelInference(model_path)
            quantifier = ColonizationQuantifier(grid_size=grid_size)
            
            results = []
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(batch_files):
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Run inference
                    prediction = inference_engine.predict(temp_path, confidence_threshold)
                    
                    # Quantify colonization
                    quantification = quantifier.quantify_colonization(temp_path, prediction)
                    
                    result = {
                        "filename": uploaded_file.name,
                        "colonization_percentage": quantification["colonization_percentage"],
                        "confidence_score": prediction["confidence"],
                        "detected_features": prediction["detected_features"],
                        "analysis_timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"Error analyzing {uploaded_file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(batch_files))
            
            if results:
                # Save results
                results_df = pd.DataFrame(results)
                results_file = f"data/results/batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                results_df.to_csv(results_file, index=False)
                
                # Display results
                st.dataframe(results_df)
                
                # Visualization
                fig = px.histogram(results_df, x="colonization_percentage", 
                                title="Distribution of Colonization Percentages")
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"Analysis complete! Results saved to: {results_file}")

def results_export_page():
    st.header("📊 Results & Export")
    
    # List available results
    result_files = [f for f in os.listdir("data/results") if f.endswith('.csv')]
    
    if not result_files:
        st.warning("No analysis results found. Please run batch analysis first.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Available Results")
        selected_result = st.selectbox("Choose result file:", result_files)
        
        if selected_result:
            result_path = os.path.join("data/results", selected_result)
            df = pd.read_csv(result_path)
            
            st.dataframe(df)
            
            # Statistics
            st.subheader("Summary Statistics")
            stats = df["colonization_percentage"].describe()
            st.write(stats)
    
    with col2:
        st.subheader("Export Options")
        
        if selected_result:
            # CSV export
            if st.button("Download CSV"):
                with open(result_path, 'rb') as f:
                    st.download_button(
                        label="Download CSV file",
                        data=f.read(),
                        file_name=selected_result,
                        mime='text/csv'
                    )
            
            # PDF report generation
            if st.button("Generate PDF Report"):
                st.info("PDF report generation would be implemented here using reportlab")
            
            # Visualization options
            st.subheader("Visualizations")
            
            chart_type = st.selectbox(
                "Chart type:",
                ["Histogram", "Box Plot", "Scatter Plot", "Time Series"]
            )
            
            if chart_type == "Histogram":
                fig = px.histogram(df, x="colonization_percentage", 
                                title="Colonization Distribution")
            elif chart_type == "Box Plot":
                fig = px.box(df, y="colonization_percentage", 
                            title="Colonization Box Plot")
            elif chart_type == "Scatter Plot":
                fig = px.scatter(df, x="confidence_score", y="colonization_percentage",
                               title="Confidence vs Colonization")
            
            st.plotly_chart(fig, use_container_width=True)

def explainability_page():
    st.header("🔍 Model Explainability (Grad-CAM)")
    
    # Check for trained model
    model_files = [f for f in os.listdir("models") if f.endswith('.pth')]
    
    if not model_files:
        st.warning("No trained models found. Please train a model first.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image for Explanation")
        
        selected_model = st.selectbox("Choose model:", model_files)
        
        uploaded_file = st.file_uploader(
            "Upload image for Grad-CAM analysis",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif']
        )
        
        if uploaded_file:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.subheader("Grad-CAM Visualization")
        
        if uploaded_file and st.button("Generate Grad-CAM"):
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Initialize Grad-CAM visualizer
                model_path = os.path.join("models", selected_model)
                gradcam_viz = GradCAMVisualizer(model_path)
                
                # Generate Grad-CAM
                gradcam_image = gradcam_viz.generate_gradcam(temp_path)
                
                # Display Grad-CAM
                st.image(gradcam_image, caption="Grad-CAM Heatmap", use_column_width=True)
                
                st.success("Grad-CAM visualization generated successfully!")
                
                # Clean up
                os.remove(temp_path)
                
            except Exception as e:
                st.error(f"Error generating Grad-CAM: {str(e)}")

if __name__ == "__main__":
    main()
