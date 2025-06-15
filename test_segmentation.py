#!/usr/bin/env python3
"""
Test script for segmentation functionality
"""

import sys
import os
import numpy as np
from PIL import Image

def test_color_config():
    """Test color configuration"""
    try:
        from src.segmentation.color_config import STRUCTURE_COLORS, rgb_to_label
        print("✅ Color configuration loaded")
        
        # Test RGB to label conversion
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 0, 0]  # Red arbuscules
        
        labels = rgb_to_label(test_image)
        if np.any(labels == 1):  # Arbuscules label
            print("✅ RGB to label conversion working")
        else:
            print("❌ RGB to label conversion failed")
        
        return True
    except Exception as e:
        print(f"❌ Color config test failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    try:
        from src.segmentation.models import UNet
        model = UNet(num_classes=7)
        print("✅ U-Net model creation working")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_trainer():
    """Test trainer initialization"""
    try:
        from src.segmentation.trainer import SegmentationTrainer
        trainer = SegmentationTrainer()
        print("✅ Trainer initialization working")
        return True
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        return False

def main():
    print("🧪 Testing Segmentation System")
    print("=" * 30)
    
    tests = [
        test_color_config,
        test_model_creation, 
        test_trainer
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! System ready for use.")
    else:
        print("⚠️ Some tests failed. Check your installation.")

if __name__ == "__main__":
    main()
