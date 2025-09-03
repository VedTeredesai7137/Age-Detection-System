#!/usr/bin/env python3
"""
Simple test script to verify CPU-only mode works
"""

import os
import sys

# Force CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Testing CPU-only TensorFlow configuration...")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    # Configure for CPU-only
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Test basic operations
    with tf.device('/CPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = tf.add(a, b)
        print(f"✓ Basic TensorFlow operation successful: {c.numpy()}")
    
    print("✓ TensorFlow CPU-only mode configured successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    from demo.SSRNET_model import SSR_net, SSR_net_general
    print("✓ SSR-NET models imported successfully")
    
    # Test model creation
    print("Creating age model...")
    age_model = SSR_net(64, [3, 3, 3], 1.0, 1.0)()
    print("✓ Age model created successfully")
    
    print("Creating gender model...")
    gender_model = SSR_net_general(64, [3, 3, 3], 1.0, 1.0)()
    print("✓ Gender model created successfully")
    
    print("🎉 All tests passed! CPU-only mode is working correctly.")
    
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)
