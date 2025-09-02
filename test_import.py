#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""
try:
    print("Testing imports...")
    
    # Test basic imports
    import cv2
    print("✅ OpenCV imported successfully")
    
    import numpy as np
    print("✅ NumPy imported successfully")
    
    import tensorflow as tf
    print("✅ TensorFlow imported successfully")
    
    import keras
    print("✅ Keras imported successfully")
    
    from mtcnn import MTCNN
    print("✅ MTCNN imported successfully")
    
    from PIL import Image
    print("✅ Pillow imported successfully")
    
    # Test SSR-NET model import
    from demo.SSRNET_model import SSR_net, SSR_net_general
    print("✅ SSR-NET model imported successfully")
    
    print("\n🎉 All imports successful! Ready for deployment.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    exit(1)
