#!/usr/bin/env python3
"""
Launch script for the segmentation system
"""

import subprocess
import sys
import os

def test_imports():
    """Test critical imports"""
    try:
        from src.segmentation.color_config import STRUCTURE_COLORS
        from src.segmentation.trainer import SegmentationTrainer
        print("✅ Segmentation modules working")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    print("🎨 Mycorrhizal Segmentation System Launcher")
    print("=" * 45)
    
    # Test imports
    if not test_imports():
        print("❌ Please run migration first: python migrate_to_segmentation.py")
        return
    
    # Launch app
    try:
        print("🚀 Launching segmentation system...")
        subprocess.run([
            "streamlit", "run", "app_segmentation.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except FileNotFoundError:
        print("❌ Streamlit not found. Install with: pip install streamlit")

if __name__ == "__main__":
    main()
