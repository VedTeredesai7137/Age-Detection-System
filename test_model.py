#!/usr/bin/env python3
"""
Test script to verify SSR-NET model loading works
"""
try:
    print("Testing SSR-NET model loading...")
    
    from demo.SSRNET_model import SSR_net, SSR_net_general
    
    # Test age model
    print("Creating age model...")
    age_model = SSR_net(64, [3, 3, 3], 1.0, 1.0)()
    print("✅ Age model created successfully")
    
    # Test gender model
    print("Creating gender model...")
    gender_model = SSR_net_general(64, [3, 3, 3], 1.0, 1.0)()
    print("✅ Gender model created successfully")
    
    print("\n🎉 All models created successfully! Ready for deployment.")
    
except Exception as e:
    print(f"❌ Model creation error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
