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
                st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    with col2:
        st.subheader("Manual Annotation")
        
        # List available images
        image_files = [f for f in os.listdir("data/raw") 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))] if os.path.exists("data/raw") else []
        
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
                
                # Feature detection
                detected_features = st.multiselect(
                    "Manually detected features:",
                    ["Arbuscules", "Vesicles", "Hyphae", "Spores", "Entry points"]
                )
                
                if st.button("Save Annotation"):
                    annotation_data = {
                        "image": selected_image,
                        "annotation_type": annotation_type,
                        "colonization_percentage": colonization_percentage,
                        "detected_features": detected_features,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Save annotation
                    annotation_file = os.path.join("data/annotations", f"{selected_image}_annotation.json")
                    with open(annotation_file, 'w') as f:
                        json.dump(annotation_data, f, indent=2)
                    
                    st.success("✅ Annotation saved!")
                    st.session_state.annotation_data[selected_image] = annotation_data
        else:
            st.info("👆 Upload images first to start annotating")

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
        
        if st.button("🚀 Start Training", type="primary"):
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
                
                # Save model
                model_path = f"models/mycorrhizal_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                trainer.save_model(model_path)
                
                st.success(f"🎉 Model trained successfully! Saved to: {model_path}")
                st.session_state.model_trained = True
                
                # Final metrics display
                final_metrics = trainer.get_training_metrics()
                if final_metrics:
                    metrics_df = pd.DataFrame(final_metrics)
                    fig = px.line(metrics_df, x='epoch', y=['train_loss', 'val_loss'], 
                                title="Final Training Progress")
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
                    ax.set_title('Confusion Matrix')
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
        selected_model = st.selectbox("Choose trained model:", model_files)
        
        # Model info
        if selected_model:
            model_path = os.path.join("models", selected_model)
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                model_info = {
                    "Model Type": checkpoint.get('model_type', 'Unknown'),
                    "Training Date": selected_model.split('_')[-1].replace('.pth', ''),
                    "File Size": f"{os.path.getsize(model_path) / (1024*1024):.1f} MB"
                }
                for key, value in model_info.items():
                    st.write(f"**{key}:** {value}")
            except:
                st.warning("Could not load model info")
        
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
                # Save results
                results_df = pd.DataFrame(results)
                results_file = f"data/results/ai_batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
                metric_col3.metric("AI Accuracy", f"{(high_confidence_count/total_count)*100:.1f}%")
                
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
                    
                    # Generate Grad-CAM
                    with st.spinner("🔄 Generating AI explanation..."):
                        gradcam_image = gradcam_viz.generate_gradcam(st.session_state.current_temp_path)
                    
                    # Display Grad-CAM
                    st.image(gradcam_image, caption="Grad-CAM: What AI Sees", use_column_width=True)
                    
                    st.success("✅ Grad-CAM visualization generated!")
                    
                    # Attention analysis
                    attention_analysis = gradcam_viz.analyze_attention_regions(
                        st.session_state.current_temp_path, 
                        gradcam_image
                    )
                    
                    st.subheader("📊 Attention Analysis")
                    st.write(f"**High Attention Regions:** {attention_analysis['num_attention_regions']}")
                    st.write(f"**Total Attention Area:** {attention_analysis['attention_percentage']:.1f}%")
                    
                    if attention_analysis['top_regions']:
                        st.write("**Top Attention Regions:**")
                        for i, region in enumerate(attention_analysis['top_regions'][:3]):
                            st.write(f"Region {i+1}: Center({region['center'][0]}, {region['center'][1]}), "
                                   f"Attention: {region['avg_attention']:.3f}")
                    
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
                        
                        gradcam_image = gradcam_viz.generate_gradcam(
                            st.session_state.current_temp_path, 
                            target_class=target_idx
                        )
                        
                        st.image(gradcam_image, caption=f"Grad-CAM for: {target_class}", 
                               use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Class-specific Grad-CAM failed: {e}")

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
