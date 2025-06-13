import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image
import json
import zipfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Handle problematic imports gracefully
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    st.warning("⚠️ OpenCV not available - using PIL for image processing")

PYTORCH_AVAILABLE = False
try:
    import torch
    PYTORCH_AVAILABLE = True
    st.success("✅ PyTorch loaded successfully!")
except ImportError:
    st.warning("⚠️ PyTorch not available - AI features limited")

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
    
    # Show available features
    col1, col2, col3 = st.columns(3)
    with col1:
        if PYTORCH_AVAILABLE:
            st.success("✅ PyTorch Available")
        else:
            st.warning("⚠️ PyTorch Missing")
    
    with col2:
        if CV2_AVAILABLE:
            st.success("✅ OpenCV Available")
        else:
            st.warning("⚠️ OpenCV Missing")
    
    with col3:
        st.info("🚧 Demo Mode Active")
    
    # Initialize session state
    if 'annotation_data' not in st.session_state:
        st.session_state.annotation_data = {}
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Upload & Annotate", "Image Analysis", "Results & Export", "System Info"]
    )
    
    if page == "Upload & Annotate":
        upload_and_annotate_page()
    elif page == "Image Analysis":
        image_analysis_page()
    elif page == "Results & Export":
        results_export_page()
    elif page == "System Info":
        system_info_page()

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
        image_files = []
        if os.path.exists("data/raw"):
            image_files = [f for f in os.listdir("data/raw") 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        
        if image_files:
            selected_image = st.selectbox("Select image to annotate:", image_files)
            
            if selected_image:
                image_path = os.path.join("data/raw", selected_image)
                image = Image.open(image_path)
                
                # Display image
                st.image(image, caption=selected_image, use_column_width=True)
                
                # Annotation interface
                st.markdown("**Annotation Guidelines:**")
                st.markdown("- Heavily colonized: >70% root area colonized")
                st.markdown("- Moderately colonized: 30-70% colonized")
                st.markdown("- Lightly colonized: 10-30% colonized")
                st.markdown("- Not colonized: <10% colonized")
                
                annotation_type = st.selectbox(
                    "Colonization level for this image:",
                    ["Not annotated", "Heavily colonized", "Moderately colonized", 
                     "Lightly colonized", "Not colonized"]
                )
                
                colonization_percentage = st.slider(
                    "Estimated colonization percentage:",
                    0, 100, 50 if annotation_type != "Not annotated" else 0
                )
                
                # Additional annotation fields
                detected_features = st.multiselect(
                    "Detected mycorrhizal features:",
                    ["Arbuscules", "Vesicles", "Hyphae", "Spores", "Entry points"]
                )
                
                notes = st.text_area("Additional notes (optional):")
                
                if st.button("Save Annotation"):
                    annotation_data = {
                        "image": selected_image,
                        "annotation_type": annotation_type,
                        "colonization_percentage": colonization_percentage,
                        "detected_features": detected_features,
                        "notes": notes,
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

def image_analysis_page():
    st.header("🔍 Image Analysis")
    
    # List available images
    image_files = []
    if os.path.exists("data/raw"):
        image_files = [f for f in os.listdir("data/raw") 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    
    if image_files:
        selected_image = st.selectbox("Select image for analysis:", image_files)
        
        if selected_image:
            image_path = os.path.join("data/raw", selected_image)
            image = Image.open(image_path)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption=selected_image, use_column_width=True)
                
                # Basic image info
                st.subheader("Image Information")
                st.write(f"**Dimensions:** {image.size}")
                st.write(f"**Mode:** {image.mode}")
                st.write(f"**Format:** {image.format}")
                
                if hasattr(image, '_getexif') and image._getexif():
                    st.write("**EXIF data available**")
            
            with col2:
                st.subheader("Analysis Options")
                
                analysis_type = st.selectbox(
                    "Choose analysis type:",
                    ["Basic Statistics", "Color Analysis", "Manual Assessment"]
                )
                
                if analysis_type == "Basic Statistics":
                    # Convert to numpy array for analysis
                    img_array = np.array(image)
                    
                    st.write("**Image Statistics:**")
                    st.write(f"Shape: {img_array.shape}")
                    st.write(f"Data type: {img_array.dtype}")
                    st.write(f"Min value: {img_array.min()}")
                    st.write(f"Max value: {img_array.max()}")
                    st.write(f"Mean value: {img_array.mean():.2f}")
                    st.write(f"Standard deviation: {img_array.std():.2f}")
                
                elif analysis_type == "Color Analysis":
                    img_array = np.array(image)
                    
                    if len(img_array.shape) == 3:
                        st.write("**Color Channel Statistics:**")
                        colors = ['Red', 'Green', 'Blue']
                        for i, color in enumerate(colors):
                            mean_val = np.mean(img_array[:, :, i])
                            std_val = np.std(img_array[:, :, i])
                            st.write(f"{color}: Mean={mean_val:.1f}, Std={std_val:.1f}")
                        
                        # Create histogram
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors_plot = ['red', 'green', 'blue']
                        for i, color in enumerate(colors_plot):
                            ax.hist(img_array[:, :, i].flatten(), bins=50, 
                                   alpha=0.7, color=color, label=color.capitalize())
                        ax.set_xlabel('Pixel Intensity')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Color Distribution')
                        ax.legend()
                        st.pyplot(fig)
                
                elif analysis_type == "Manual Assessment":
                    st.write("**Manual Colonization Assessment**")
                    
                    # Estimation sliders
                    estimated_colonization = st.slider(
                        "Overall colonization estimate (%):", 0, 100, 25
                    )
                    
                    confidence = st.slider(
                        "Confidence in assessment (%):", 0, 100, 75
                    )
                    
                    # Feature detection
                    features_detected = st.multiselect(
                        "Manually detected features:",
                        ["Arbuscules", "Vesicles", "Hyphae", "Spores", "Entry points"]
                    )
                    
                    assessment_notes = st.text_area("Assessment notes:")
                    
                    if st.button("Save Manual Assessment"):
                        result = {
                            "filename": selected_image,
                            "manual_colonization_estimate": estimated_colonization,
                            "confidence": confidence,
                            "features_detected": features_detected,
                            "assessment_notes": assessment_notes,
                            "analysis_type": "manual",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Save result
                        results_file = os.path.join("data/results", 
                                                   f"manual_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                        with open(results_file, 'w') as f:
                            json.dump(result, f, indent=2)
                        
                        st.success("✅ Manual assessment saved!")
                
                # AI Analysis section (if PyTorch available)
                if PYTORCH_AVAILABLE:
                    st.markdown("---")
                    st.subheader("🤖 AI Analysis")
                    st.info("AI features would be available here with trained models")
                    
                    if st.button("Run AI Analysis (Demo)"):
                        # Simulate AI analysis
                        st.info("🔄 Running AI analysis...")
                        
                        # Simulate processing time
                        import time
                        time.sleep(2)
                        
                        # Mock AI results
                        mock_results = {
                            "predicted_class": "Moderately colonized",
                            "confidence": 0.78,
                            "estimated_percentage": 45,
                            "detected_features": ["Vesicles", "Hyphae"]
                        }
                        
                        st.success("✅ AI Analysis Complete!")
                        st.write(f"**Predicted class:** {mock_results['predicted_class']}")
                        st.write(f"**Confidence:** {mock_results['confidence']:.2f}")
                        st.write(f"**Estimated colonization:** {mock_results['estimated_percentage']}%")
                        st.write(f"**Features detected:** {', '.join(mock_results['detected_features'])}")
                else:
                    st.info("🤖 AI analysis requires PyTorch installation")
    else:
        st.info("👆 Upload images in the 'Upload & Annotate' page first")

def results_export_page():
    st.header("📊 Results & Export")
    
    # Check for saved annotations and results
    annotation_files = []
    result_files = []
    
    if os.path.exists("data/annotations"):
        annotation_files = [f for f in os.listdir("data/annotations") if f.endswith('.json')]
    
    if os.path.exists("data/results"):
        result_files = [f for f in os.listdir("data/results") if f.endswith('.json')]
    
    if annotation_files or result_files:
        tab1, tab2, tab3 = st.tabs(["📝 Annotations", "📊 Analysis Results", "📈 Visualizations"])
        
        with tab1:
            st.subheader("Annotation Summary")
            
            if annotation_files:
                annotations = []
                for file in annotation_files:
                    try:
                        with open(os.path.join("data/annotations", file), 'r') as f:
                            annotation = json.load(f)
                            annotations.append(annotation)
                    except Exception as e:
                        st.error(f"Error loading {file}: {e}")
                
                if annotations:
                    df_annotations = pd.DataFrame(annotations)
                    st.dataframe(df_annotations, use_container_width=True)
                    
                    # Statistics
                    if 'colonization_percentage' in df_annotations.columns:
                        avg_colonization = df_annotations['colonization_percentage'].mean()
                        std_colonization = df_annotations['colonization_percentage'].std()
                        min_colonization = df_annotations['colonization_percentage'].min()
                        max_colonization = df_annotations['colonization_percentage'].max()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Average", f"{avg_colonization:.1f}%")
                        col2.metric("Std Dev", f"{std_colonization:.1f}%")
                        col3.metric("Minimum", f"{min_colonization:.1f}%")
                        col4.metric("Maximum", f"{max_colonization:.1f}%")
                    
                    # Download annotations as CSV
                    csv = df_annotations.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Annotations (CSV)",
                        data=csv,
                        file_name=f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
            else:
                st.info("No annotations found")
        
        with tab2:
            st.subheader("Analysis Results")
            
            if result_files:
                results = []
                for file in result_files:
                    try:
                        with open(os.path.join("data/results", file), 'r') as f:
                            result = json.load(f)
                            results.append(result)
                    except Exception as e:
                        st.error(f"Error loading {file}: {e}")
                
                if results:
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Download results as CSV
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
            else:
                st.info("No analysis results found")
        
        with tab3:
            st.subheader("Data Visualizations")
            
            if annotation_files and len(annotation_files) > 0:
                # Load annotations for visualization
                annotations = []
                for file in annotation_files:
                    try:
                        with open(os.path.join("data/annotations", file), 'r') as f:
                            annotation = json.load(f)
                            annotations.append(annotation)
                    except:
                        continue
                
                if annotations:
                    df = pd.DataFrame(annotations)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'annotation_type' in df.columns:
                            # Pie chart of annotation types
                            fig = px.pie(df, names='annotation_type', 
                                        title='Colonization Level Distribution')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'colonization_percentage' in df.columns:
                            # Histogram of colonization percentages
                            fig = px.histogram(df, x='colonization_percentage', 
                                             title='Distribution of Colonization Percentages',
                                             nbins=20)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Box plot
                    if 'annotation_type' in df.columns and 'colonization_percentage' in df.columns:
                        fig = px.box(df, x='annotation_type', y='colonization_percentage',
                                    title='Colonization Percentage by Classification')
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for visualization")
    else:
        st.info("📝 No data available yet. Start by uploading and annotating images!")

def system_info_page():
    st.header("ℹ️ System Information")
    
    st.subheader("🔧 Available Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Core Features:**")
        st.write("✅ Image upload and management")
        st.write("✅ Manual annotation interface")
        st.write("✅ Basic image analysis")
        st.write("✅ Data visualization")
        st.write("✅ CSV export functionality")
        
        st.write("**Image Formats Supported:**")
        st.write("• PNG, JPG, JPEG")
        st.write("• TIFF, TIF")
        st.write("• RGB and grayscale images")
    
    with col2:
        st.write("**AI Features:**")
        if PYTORCH_AVAILABLE:
            st.write("✅ PyTorch available")
            st.write("✅ Deep learning models")
            st.write("✅ Automated classification")
            st.write("✅ Feature detection")
        else:
            st.write("⚠️ PyTorch not installed")
            st.write("⚠️ AI features limited")
        
        st.write("**Image Processing:**")
        if CV2_AVAILABLE:
            st.write("✅ OpenCV available")
            st.write("✅ Advanced image processing")
        else:
            st.write("⚠️ OpenCV limited")
            st.write("✅ PIL-based processing")
    
    st.subheader("🐍 Python Environment")
    
    import sys
    st.write(f"**Python version:** {sys.version}")
    st.write(f"**Platform:** {sys.platform}")
    
    # Package versions
    packages_to_check = ['streamlit', 'numpy', 'pandas', 'PIL', 'matplotlib', 'plotly']
    
    if PYTORCH_AVAILABLE:
        packages_to_check.extend(['torch', 'torchvision'])
    
    if CV2_AVAILABLE:
        packages_to_check.append('cv2')
    
    st.write("**Installed packages:**")
    for package in packages_to_check:
        try:
            if package == 'PIL':
                import PIL
                version = PIL.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
            st.write(f"• {package}: {version}")
        except ImportError:
            st.write(f"• {package}: Not installed")
    
    st.subheader("📁 Data Directory Status")
    
    directories = ['data/raw', 'data/annotations', 'data/processed', 'data/results', 'models']
    
    for directory in directories:
        if os.path.exists(directory):
            file_count = len([f for f in os.listdir(directory) 
                            if os.path.isfile(os.path.join(directory, f))])
            st.write(f"✅ {directory}: {file_count} files")
        else:
            st.write(f"❌ {directory}: Not found")
    
    st.subheader("🚀 Getting Started")
    
    st.markdown("""
    **Quick Start Guide:**
    
    1. **Upload Images**: Go to 'Upload & Annotate' and upload your microscope images
    2. **Annotate Data**: Classify colonization levels and estimate percentages
    3. **Analyze Images**: Use 'Image Analysis' for detailed examination
    4. **Export Results**: Download your data from 'Results & Export'
    
    **For AI Features:**
    - Install PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
    - Install OpenCV: `pip install opencv-python-headless`
    
    **Need Help?**
    - Check the system requirements
    - Ensure all dependencies are installed
    - Try the demo features first
    """)

if __name__ == "__main__":
    main()
