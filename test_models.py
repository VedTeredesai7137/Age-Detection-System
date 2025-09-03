#!/usr/bin/env python3
"""
Test script to verify model loading and basic functionality
Run this before deployment to ensure everything works
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        import cv2
        logger.info("✓ OpenCV imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        logger.info("✓ NumPy imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        import tensorflow as tf
        logger.info(f"✓ TensorFlow imported successfully (version: {tf.__version__})")
    except Exception as e:
        logger.error(f"✗ Failed to import TensorFlow: {e}")
        return False
    
    try:
        import keras
        logger.info(f"✓ Keras imported successfully (version: {keras.__version__})")
    except Exception as e:
        logger.error(f"✗ Failed to import Keras: {e}")
        return False
    
    try:
        from mtcnn import MTCNN
        logger.info("✓ MTCNN imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import MTCNN: {e}")
        return False
    
    try:
        from demo.SSRNET_model import SSR_net, SSR_net_general
        logger.info("✓ SSR-NET models imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import SSR-NET models: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if models can be created"""
    logger.info("Testing model creation...")
    
    try:
        from demo.SSRNET_model import SSR_net, SSR_net_general
        
        # Test age model creation
        logger.info("Creating age model...")
        age_model = SSR_net(64, [3, 3, 3], 1.0, 1.0)()
        logger.info("✓ Age model created successfully")
        
        # Test gender model creation
        logger.info("Creating gender model...")
        gender_model = SSR_net_general(64, [3, 3, 3], 1.0, 1.0)()
        logger.info("✓ Gender model created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_file_structure():
    """Test if required files and directories exist"""
    logger.info("Testing file structure...")
    
    required_files = [
        'demo/SSRNET_model.py',
        'demo/lbpcascade_frontalface_improved.xml',
        'app.py',
        'requirements.txt'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            logger.info(f"✓ {file_path} exists")
        else:
            logger.error(f"✗ {file_path} missing")
            return False
    
    # Check for model weights
    model_dirs = [
        'pre-trained/imdb/ssrnet_3_3_3_64_1.0_1.0',
        'pre-trained/wiki/ssrnet_3_3_3_64_1.0_1.0',
        'pre-trained/morph2/ssrnet_3_3_3_64_1.0_1.0'
    ]
    
    found_weights = False
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            logger.info(f"✓ Model directory found: {model_dir}")
            # Check for .h5 files
            h5_files = [f for f in os.listdir(model_dir) if f.endswith('.h5') and not f.startswith('history_')]
            if h5_files:
                logger.info(f"✓ Found model weights: {h5_files[0]}")
                found_weights = True
                break
    
    if not found_weights:
        logger.warning("⚠ No model weights found - this will cause issues during deployment")
    
    return True

def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Age Detection SSR-NET - Pre-deployment Test")
    logger.info("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Model Creation", test_model_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} Test ---")
        if test_func():
            logger.info(f"✓ {test_name} test PASSED")
            passed += 1
        else:
            logger.error(f"✗ {test_name} test FAILED")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Ready for deployment.")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
