# Mycorrhizal Colonization Detection System

An AI-powered web application for automated detection and quantification of mycorrhizal colonization in plant root microscope images.

## Features

- **Image Upload & Annotation**: Upload microscope images and manually annotate colonization levels
- **Deep Learning Training**: Train CNN models (ResNet, EfficientNet) for colonization detection
- **Batch Analysis**: Process multiple images automatically with confidence scoring
- **Quantification Methods**: Multiple approaches including gridline intersection and area percentage
- **Results Export**: Export results in CSV format with detailed metrics
- **Model Explainability**: Grad-CAM visualizations to understand model decisions

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/mycorrhizal-detection.git
cd mycorrhizal-detection

# Build and run with Docker
docker-compose up --build

# Access the application at http://localhost:8501
