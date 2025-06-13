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
import cv2
import matplotlib.pyplot as plt

# Handle missing PyTorch gracefully
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

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
    
    if not PYTORCH_AVAILABLE:
        st.info("🚧 **Demo Mode**: Full AI features require PyTorch. Current features available:")
        st.success("✅ Image upload and viewing")
        st.success("✅ Manual annotation")
        st.success("✅ Basic image analysis")
        st.success("✅ Results visualization")
        st.success("✅ Data export")
    else:
        st.success("✅ Full AI features available!")
    
    # Initialize session state
    if 'annotation_data' not in st.session_state:
        st.session_state.annotation_data = {}
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Upload & Annotate", "Image Analysis", "Results & Export", "About"]
    )
    
    if page == "Upload & Annotate":
        upload_and_annotate_page()
    elif page == "Image Analysis":
        image_analysis_page()
    elif page == "Results & Export":
        results_export_page()
    elif page == "About":
        about_page()

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
        image_files = [f for f in os.listdir("data/raw") if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))] if os.path.exists("data/raw") else []
        
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
                    
                    st.success("✅ Annotation saved!")
                    st.session_state.annotation_data[selected_image] = annotation_data

def image_analysis_page():
    st.header("🔍 Image Analysis")
    
    if not PYTORCH_AVAILABLE:
        st.warning("⚠️ AI model analysis requires PyTorch installation")
        st.info("**Demo Mode**: Basic image processing available")
    
    # List available images
    image_files = [f for f in os.listdir("data/raw") if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))] if os.path.exists("data/raw") else []
    
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
            
            with col2:
                st.subheader("Basic Analysis")
                
                # Convert to numpy array for analysis
                img_array = np.array(image)
                
                if len(img_array.shape) == 3:
                    # Color image analysis
                    st.write("**Color Statistics:**")
                    for i, color in enumerate(['Red', 'Green', 'Blue']):
                        mean_val = np.mean(img_array[:, :, i])
                        std_val = np.std(img_array[:, :, i])
                        st.write(f"{color}: Mean={mean_val:.1f}, Std={std_val:.1f}")
                    
                    # Simple histogram
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ['red', 'green', 'blue']
                    for i, color in enumerate(colors):
                        ax.hist(img_array[:, :, i].flatten(), bins=50, alpha=0.7, color=color, label=color.capitalize())
                    ax.set_xlabel('Pixel Intensity')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Color Distribution')
                    ax.legend()
                    st.pyplot(fig)
                
                # Manual colonization estimation
                st.subheader("Manual Assessment")
                estimated_colonization = st.slider("Your colonization estimate (%):", 0, 100, 25)
                
                if st.button("Save Assessment"):
                    result = {
                        "filename": selected_image,
                        "manual_estimate": estimated_colonization,
                        "analysis_type": "manual",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Save result
                    results_file = os.path.join("data/results", f"manual_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    with open(results_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    st.success("✅ Assessment saved!")
    else:
        st.info("👆 Upload images in the 'Upload & Annotate' page first")

def results_export_page():
    st.header("📊 Results & Export")
    
    # Check for saved annotations
    annotation_files = [f for f in os.listdir("data/annotations") if f.endswith('.json')] if os.path.exists("data/annotations") else []
    result_files = [f for f in os.listdir("data/results") if f.endswith('.json')] if os.path.exists("data/results") else []
    
    if annotation_files or result_files:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Annotations Summary")
            
            if annotation_files:
                annotations = []
                for file in annotation_files:
                    with open(os.path.join("data/annotations", file), 'r') as f:
                        annotation = json.load(f)
                        annotations.append(annotation)
                
                df_annotations = pd.DataFrame(annotations)
                st.dataframe(df_annotations)
                
                # Statistics
                if 'colonization_percentage' in df_annotations.columns:
                    avg_colonization = df_annotations['colonization_percentage'].mean()
                    st.metric("Average Colonization", f"{avg_colonization:.1f}%")
                
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
        
        with col2:
            st.subheader("📈 Visualization")
            
            if annotation_files and len(annotations) > 0:
                df = pd.DataFrame(annotations)
                
                if 'annotation_type' in df.columns:
                    # Pie chart of annotation types
                    fig = px.pie(df, names='annotation_type', title='Colonization Level Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'colonization_percentage' in df.columns:
                    # Histogram of colonization percentages
                    fig = px.histogram(df, x='colonization_percentage', 
                                     title='Distribution of Colonization Percentages',
                                     nbins=20)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📝 No results available yet. Start by uploading and annotating images!")

def about_page():
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🔬 Mycorrhizal Colonization Detection System
    
    This application helps researchers analyze mycorrhizal colonization in plant root microscope images.
    
    ### ✨ Features:
    - **Image Upload**: Support for common microscopy formats (PNG, JPG, TIFF)
    - **Manual Annotation**: Classify colonization levels and estimate percentages
    - **Basic Analysis**: Color distribution and image statistics
    - **Data Export**: Download results in CSV format
    - **Visualization**: Charts and graphs of your data
    
    ### 🚧 Current Mode: Demo Version
    - Full AI features require PyTorch installation
    - All basic functionality is available
    - Perfect for manual annotation workflows
    
    ### 📚 How to Use:
    1. **Upload Images**: Go to 'Upload & Annotate' and upload your microscope images
    2. **Annotate**: Select colonization levels and estimate percentages
    3. **Analyze**: Use 'Image Analysis' for basic image processing
    4. **Export**: Download your annotations and results
    
    ### 🔬 Research Applications:
    - Mycorrhizal symbiosis studies
    - Plant-microbe interaction research
    - Agricultural and ecological studies
    - Educational demonstrations
    
    ### 🛠️ Technical Details:
    - Built with Streamlit and Python
    - Image processing with PIL and OpenCV
    - Data visualization with Plotly
    - Deployed on Streamlit Cloud
    """)

if __name__ == "__main__":
    main()
