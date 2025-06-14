#!/usr/bin/env python3
"""
Simple launch script for GitHub Codespaces
Maintains full AI functionality while optimizing for Codespaces
"""

import os
import sys
import subprocess

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

def install_dependencies():
    """Install dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Use your original requirements.txt (with fixed syntax)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def launch_app():
    """Launch the Streamlit application with optimized settings"""
    print("🚀 Launching Mycorrhizal Detection System...")
    
    try:
        # Launch with Codespaces-friendly settings (but keep all AI features)
        cmd = [
            "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ]
        
        print("🔬 Access your app at the forwarded port 8501")
        print("📱 Look for the popup or check the PORTS tab in VS Code")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"\n❌ Launch failed: {e}")

def main():
    """Main launch function"""
    print("🔬 Mycorrhizal Colonization Detection System")
    print("=" * 50)
    
    # Setup
    setup_directories()
    
    # Install dependencies (keeps your original PyTorch 2.4.0)
    if not install_dependencies():
        print("Try manual installation: pip install -r requirements.txt")
        return
    
    # Launch with full AI functionality
    launch_app()

if __name__ == "__main__":
    main()
