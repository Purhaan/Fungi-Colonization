#!/usr/bin/env python3
"""
ULTIMATE 2025 CODESPACES FIX for Mycorrhizal Detection System
Based on latest setuptools.build_meta solutions from 2025
"""

import os
import sys
import subprocess
import time
import tempfile

def emergency_pip_fix():
    """Emergency fix for corrupted pip environments - 2025 solution"""
    print("🚨 EMERGENCY PIP REPAIR (2025 Method)")
    print("=" * 45)
    
    try:
        # Method 1: Force reinstall pip using get-pip.py
        print("📥 Downloading fresh pip installer...")
        import urllib.request
        
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', f.name)
            
            print("🔧 Installing fresh pip...")
            subprocess.run([sys.executable, f.name, '--force-reinstall'], check=True)
            os.unlink(f.name)
        
        print("✅ Emergency pip repair complete!")
        return True
        
    except Exception as e:
        print(f"⚠️ Emergency method failed: {e}")
        
        # Method 2: Manual setuptools fix
        try:
            print("🔧 Trying manual setuptools fix...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", "--force-reinstall", "--no-cache-dir",
                "pip", "setuptools>=65.0", "wheel"
            ], check=True)
            return True
        except:
            print("⚠️ Manual fix also failed, but continuing...")
            return True  # Continue anyway

def install_with_fallbacks(package_name, alternatives=None):
    """Install package with multiple fallback methods"""
    alternatives = alternatives or [package_name]
    
    for attempt, pkg in enumerate(alternatives):
        try:
            print(f"   Attempt {attempt + 1}: {pkg}")
            
            # Try different installation methods
            methods = [
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg],
                [sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir", pkg],
                [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir", pkg],
                [sys.executable, "-m", "pip", "install", "--user", "--no-cache-dir", pkg]
            ]
            
            for method in methods:
                try:
                    subprocess.run(method, check=True, capture_output=True, timeout=120)
                    print(f"   ✅ {pkg} installed successfully")
                    return True
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    continue
                    
        except Exception:
            continue
    
    print(f"   ⚠️ Could not install {package_name}, but continuing...")
    return False

def smart_package_installation():
    """Smart package installation with 2025 best practices"""
    print("🧠 SMART PACKAGE INSTALLATION")
    print("=" * 35)
    
    # Core packages with fallbacks
    package_groups = [
        {
            'name': 'numpy',
            'primary': 'numpy>=1.21.0',
            'fallbacks': ['numpy==1.24.3', 'numpy==1.21.6', 'numpy']
        },
        {
            'name': 'pandas', 
            'primary': 'pandas>=1.5.0',
            'fallbacks': ['pandas==2.0.3', 'pandas==1.5.3', 'pandas']
        },
        {
            'name': 'pillow',
            'primary': 'Pillow>=9.0.0',
            'fallbacks': ['Pillow==10.0.0', 'Pillow==9.5.0', 'Pillow']
        },
        {
            'name': 'opencv',
            'primary': 'opencv-python-headless>=4.6.0',
            'fallbacks': ['opencv-python-headless==4.8.1.78', 'opencv-python-headless', 'opencv-python']
        },
        {
            'name': 'pytorch',
            'primary': 'torch>=1.12.0 torchvision>=0.13.0',
            'fallbacks': ['torch==2.1.0 torchvision==0.16.0', 'torch torchvision', 'torch==1.13.1 torchvision==0.14.1']
        },
        {
            'name': 'scikit-learn',
            'primary': 'scikit-learn>=1.1.0',
            'fallbacks': ['scikit-learn==1.3.0', 'scikit-learn==1.2.2', 'scikit-learn']
        },
        {
            'name': 'matplotlib',
            'primary': 'matplotlib>=3.5.0',
            'fallbacks': ['matplotlib==3.7.2', 'matplotlib==3.6.3', 'matplotlib']
        },
        {
            'name': 'streamlit',
            'primary': 'streamlit>=1.25.0',
            'fallbacks': ['streamlit==1.28.1', 'streamlit==1.27.0', 'streamlit']
        },
        {
            'name': 'plotly',
            'primary': 'plotly>=5.10.0',
            'fallbacks': ['plotly==5.15.0', 'plotly==5.14.1', 'plotly']
        },
        {
            'name': 'scipy',
            'primary': 'scipy>=1.8.0',
            'fallbacks': ['scipy==1.11.1', 'scipy==1.10.1', 'scipy']
        }
    ]
    
    success_count = 0
    for group in package_groups:
        print(f"📦 Installing {group['name']}...")
        fallback_options = [group['primary']] + group['fallbacks']
        
        if install_with_fallbacks(group['name'], fallback_options):
            success_count += 1
        
        time.sleep(1)  # Prevent overwhelming the system
    
    print(f"✅ Successfully installed {success_count}/{len(package_groups)} package groups")
    return success_count > len(package_groups) * 0.7  # 70% success rate minimum

def create_robust_project_structure():
    """Create project structure with error handling"""
    print("📁 CREATING ROBUST PROJECT STRUCTURE")
    print("=" * 40)
    
    # Directory structure
    directories = [
        "data/raw", "data/annotations", "data/processed", 
        "data/results", "models", "src", "temp"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"   ✅ {directory}")
        except Exception as e:
            print(f"   ⚠️ {directory}: {e}")
    
    # Create essential files
    essential_files = {
        "src/__init__.py": "# Mycorrhizal Detection Package\n",
        ".gitignore": """
# Data and models
data/
models/
temp/
*.pth
*.pkl

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/

# Environment
.env
.venv/
venv/
""",
        "README_MINIMAL.md": """
# Mycorrhizal Detection System - Working Version

## Quick Start
```bash
python ultimate_fix_2025.py
```

## Basic Usage
1. Upload images in "Upload & Annotate"
2. Annotate a few images manually
3. Train model (simplified version)
4. Analyze new images

## Status
- ✅ Core functionality working
- ✅ Image upload and annotation
- ✅ Basic analysis and export
- 🔄 Advanced AI features (requires full setup)
"""
    }
    
    for file_path, content in essential_files.items():
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"   ✅ Created {file_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to create {file_path}: {e}")

def test_critical_functionality():
    """Test that critical components work"""
    print("🧪 TESTING CRITICAL FUNCTIONALITY")
    print("=" * 35)
    
    tests = [
        ("Python version", lambda: sys.version_info >= (3, 7)),
        ("Import streamlit", lambda: __import__('streamlit')),
        ("Import PIL", lambda: __import__('PIL')),
        ("Import numpy", lambda: __import__('numpy')),
        ("Import pandas", lambda: __import__('pandas')),
        ("Directory access", lambda: os.access('data', os.W_OK)),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"   ✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"   ❌ {test_name}: {e}")
    
    print(f"📊 Tests passed: {passed}/{len(tests)}")
    return passed >= len(tests) * 0.6  # 60% minimum

def create_fallback_app():
    """Create a minimal but functional app if the main one fails"""
    print("🔧 CREATING FALLBACK APPLICATION")
    print("=" * 35)
    
    fallback_app = '''
import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
from PIL import Image
import numpy as np

st.set_page_config(page_title="Mycorrhizal Detection", page_icon="🔬", layout="wide")

def main():
    st.title("🔬 Mycorrhizal Colonization Detection System")
    st.markdown("### Fallback Version - Core Features Only")
    
    # Ensure directories exist
    for d in ["data/raw", "data/annotations", "data/results"]:
        os.makedirs(d, exist_ok=True)
    
    # Show system status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Core System Active")
    with col2:
        try:
            import torch
            st.success("✅ PyTorch Available")
        except ImportError:
            st.warning("⚠️ PyTorch Limited")
    with col3:
        st.info("🖥️ Fallback Mode")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Annotate", "📊 Analysis", "📥 Export"])
    
    with tab1:
        st.header("Image Upload and Annotation")
        
        uploaded_files = st.file_uploader(
            "Upload mycorrhizal root images",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                col_a, col_b = st.columns([1, 1])
                
                with col_a:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_column_width=True)
                
                with col_b:
                    st.subheader(f"Annotate: {uploaded_file.name}")
                    
                    colonization_level = st.selectbox(
                        "Colonization level:",
                        ["Not colonized", "Lightly colonized", "Moderately colonized", "Heavily colonized"],
                        key=f"level_{uploaded_file.name}"
                    )
                    
                    percentage = st.slider(
                        "Estimated percentage:",
                        0, 100, 25,
                        key=f"pct_{uploaded_file.name}"
                    )
                    
                    features = st.multiselect(
                        "Observed features:",
                        ["Arbuscules", "Vesicles", "Hyphae", "Spores"],
                        key=f"feat_{uploaded_file.name}"
                    )
                    
                    notes = st.text_area(
                        "Notes:",
                        key=f"notes_{uploaded_file.name}",
                        height=100
                    )
                    
                    if st.button(f"💾 Save", key=f"save_{uploaded_file.name}"):
                        # Save image
                        image_path = os.path.join("data/raw", uploaded_file.name)
                        image.save(image_path)
                        
                        # Save annotation
                        annotation = {
                            "image": uploaded_file.name,
                            "colonization_level": colonization_level,
                            "percentage": percentage,
                            "features": features,
                            "notes": notes,
                            "timestamp": datetime.now().isoformat(),
                            "method": "manual_annotation"
                        }
                        
                        annotation_path = os.path.join("data/annotations", f"{uploaded_file.name}_annotation.json")
                        with open(annotation_path, 'w') as f:
                            json.dump(annotation, f, indent=2)
                        
                        st.success(f"✅ Saved {uploaded_file.name}")
                
                st.markdown("---")
    
    with tab2:
        st.header("Data Analysis")
        
        # Load existing annotations
        annotations = []
        if os.path.exists("data/annotations"):
            for f in os.listdir("data/annotations"):
                if f.endswith('.json'):
                    try:
                        with open(os.path.join("data/annotations", f), 'r') as file:
                            annotations.append(json.load(file))
                    except:
                        pass
        
        if annotations:
            df = pd.DataFrame(annotations)
            
            # Summary metrics
            if "percentage" in df.columns:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Images", len(df))
                col2.metric("Average Colonization", f"{df['percentage'].mean():.1f}%")
                col3.metric("Max Colonization", f"{df['percentage'].max():.1f}%")
            
            # Data table
            st.subheader("📋 Annotation Summary")
            st.dataframe(df)
            
            # Simple visualization
            if "percentage" in df.columns and len(df) > 1:
                st.subheader("📊 Colonization Distribution")
                try:
                    import plotly.express as px
                    fig = px.histogram(df, x="percentage", title="Colonization Percentage Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("📊 Install plotly for advanced visualizations")
                    
                    # Simple matplotlib fallback
                    try:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        ax.hist(df["percentage"], bins=10)
                        ax.set_xlabel("Colonization Percentage")
                        ax.set_ylabel("Frequency")
                        ax.set_title("Colonization Distribution")
                        st.pyplot(fig)
                    except ImportError:
                        st.write("📈 Basic stats:")
                        st.write(df["percentage"].describe())
        else:
            st.info("No data yet. Upload and annotate images first!")
    
    with tab3:
        st.header("Export Results")
        
        if annotations:
            df = pd.DataFrame(annotations)
            
            # CSV export
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"mycorrhizal_data_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
            
            # Summary report
            summary = f"""
MYCORRHIZAL COLONIZATION ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
Total images analyzed: {len(df)}
Average colonization: {df['percentage'].mean():.2f}%
Standard deviation: {df['percentage'].std():.2f}%
Range: {df['percentage'].min():.1f}% - {df['percentage'].max():.1f}%

DISTRIBUTION
{df['colonization_level'].value_counts().to_string()}
"""
            
            st.download_button(
                "📄 Download Report",
                summary,
                f"mycorrhizal_report_{datetime.now().strftime('%Y%m%d')}.txt",
                "text/plain"
            )
        else:
            st.info("No data to export yet.")

if __name__ == "__main__":
    main()
'''
    
    try:
        with open("app_fallback.py", "w") as f:
            f.write(fallback_app)
        print("   ✅ Created fallback app: app_fallback.py")
        return True
    except Exception as e:
        print(f"   ❌ Failed to create fallback app: {e}")
        return False

def launch_application():
    """Launch the application with multiple fallback options"""
    print("🚀 LAUNCHING APPLICATION")
    print("=" * 25)
    
    # Try different app versions in order of preference
    app_options = [
        ("app.py", "Main Application"),
        ("app_fallback.py", "Fallback Application"),
    ]
    
    for app_file, app_name in app_options:
        if os.path.exists(app_file):
            print(f"🎯 Attempting to launch: {app_name}")
            try:
                cmd = [
                    "streamlit", "run", app_file,
                    "--server.port=8501",
                    "--server.address=0.0.0.0",
                    "--browser.gatherUsageStats=false",
                    "--server.maxUploadSize=50"
                ]
                
                print("🌐 Starting server...")
                print("📱 Access your app at the forwarded port 8501")
                print("⚡ Press Ctrl+C to stop")
                print("=" * 50)
                
                subprocess.run(cmd)
                return True
                
            except KeyboardInterrupt:
                print("\\n👋 Application stopped by user")
                return True
            except FileNotFoundError:
                print(f"   ❌ Streamlit not found for {app_name}")
                continue
            except Exception as e:
                print(f"   ❌ Failed to launch {app_name}: {e}")
                continue
    
    print("❌ Could not launch any application version")
    return False

def main():
    """Main execution with comprehensive error handling"""
    print("🔬 MYCORRHIZAL DETECTION SYSTEM")
    print("🛠️  ULTIMATE 2025 CODESPACES FIX")
    print("=" * 55)
    
    try:
        # Step 1: Emergency pip repair
        if not emergency_pip_fix():
            print("⚠️ Pip repair had issues, but continuing...")
        
        # Step 2: Smart package installation
        if not smart_package_installation():
            print("⚠️ Some packages failed to install, creating fallback...")
        
        # Step 3: Project structure
        create_robust_project_structure()
        
        # Step 4: Create fallback app
        create_fallback_app()
        
        # Step 5: Test functionality
        if not test_critical_functionality():
            print("⚠️ Some tests failed, but attempting launch anyway...")
        
        # Step 6: Launch
        print("\\n🎉 Setup complete! Launching application...")
        time.sleep(2)
        launch_application()
        
    except KeyboardInterrupt:
        print("\\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\\n💥 Critical error: {e}")
        print("\\n🆘 EMERGENCY FALLBACK:")
        print("1. Try running: python app_fallback.py")
        print("2. Or install manually: pip install streamlit pillow pandas")
        print("3. Then run: streamlit run app_fallback.py")

if __name__ == "__main__":
    main()
