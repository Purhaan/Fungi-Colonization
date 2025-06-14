#!/usr/bin/env python3
"""
Comprehensive Codespaces fix and launch script
Fixes all dependency issues permanently
"""

import os
import sys
import subprocess
import time

def install_system_dependencies():
    """Install required system packages"""
    print("📦 Installing system dependencies...")
    
    system_packages = [
        "libgl1-mesa-glx",
        "libglib2.0-0", 
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "libpng16-16",
        "libfontconfig1-dev",
        "libfreetype6-dev"
    ]
    
    try:
        subprocess.run(["sudo", "apt-get", "update", "-q"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y"] + system_packages, check=True)
        print("✅ System dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install system dependencies: {e}")
        return False

def clean_python_environment():
    """Clean up conflicting Python packages"""
    print("🧹 Cleaning Python environment...")
    
    # Packages that commonly conflict
    problematic_packages = [
        "Pillow", "PIL", "plotly", "pandas", "numpy", 
        "opencv-python", "opencv-python-headless", "narwhals"
    ]
    
    try:
        # Uninstall potentially conflicting packages
        for package in problematic_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "uninstall", package, "-y"], 
                             capture_output=True)
            except:
                pass
        
        # Clear pip cache
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                      capture_output=True)
        
        print("✅ Environment cleaned")
        return True
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
        return True  # Continue anyway

def install_python_dependencies():
    """Install Python packages in correct order"""
    print("🐍 Installing Python dependencies...")
    
    # Install packages in specific order to avoid conflicts
    package_groups = [
        # Core packages first
        ["pip", "setuptools", "wheel"],
        # Base scientific packages
        ["numpy==1.24.3"],
        ["pandas==2.0.3"],
        # Image processing
        ["Pillow==10.0.0"],
        ["opencv-python-headless==4.8.1.78"],
        # ML packages
        ["torch==2.1.0", "torchvision==0.16.0"],
        ["scikit-learn==1.3.0"],
        ["scipy==1.11.1"],
        ["scikit-image==0.21.0"],
        # Visualization
        ["matplotlib==3.7.2"],
        ["seaborn==0.12.2"], 
        ["plotly==5.15.0"],
        # Web framework
        ["streamlit==1.28.1"]
    ]
    
    for group in package_groups:
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + group
            subprocess.run(cmd, check=True)
            print(f"   ✅ Installed: {', '.join(group)}")
            time.sleep(1)  # Brief pause between groups
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {group}: {e}")
            return False
    
    print("✅ All Python dependencies installed")
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "data/raw",
        "data/annotations",
        "data/processed", 
        "data/results",
        "models",
        "src"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Ensure __init__.py exists
    init_file = "src/__init__.py"
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Mycorrhizal Detection Package\n")
    
    print("✅ Directories ready")

def launch_app():
    """Launch the Streamlit application"""
    print("🚀 Launching Mycorrhizal Detection System...")
    print("🔬 The app will be available at the forwarded port 8501")
    print("📱 Look for the popup or check the PORTS tab in VS Code")
    
    try:
        cmd = [
            "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"\n❌ Launch failed: {e}")

def main():
    """Main execution function"""
    print("🔬 Mycorrhizal Colonization Detection System")
    print("=" * 50)
    print("🔧 Comprehensive Codespaces Setup & Launch")
    print("=" * 50)
    
    # Step 1: Install system dependencies
    if not install_system_dependencies():
        print("❌ Setup failed at system dependencies")
        return
    
    # Step 2: Clean environment
    clean_python_environment()
    
    # Step 3: Install Python packages
    if not install_python_dependencies():
        print("❌ Setup failed at Python dependencies")
        return
    
    # Step 4: Setup directories
    setup_directories()
    
    # Step 5: Launch app
    print("\n🎉 Setup complete! Launching application...")
    launch_app()

if __name__ == "__main__":
    main()
