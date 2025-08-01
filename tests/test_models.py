#!/usr/bin/env python3
"""
Model Test Script for Raspberry Pi 5 Face Recognition System

This script tests the TFLite models used in the face recognition system.

Usage:
    python3 tests/test_models.py
"""

import tflite_runtime.interpreter as tflite
import numpy as np
import os

def test_models():
    """Test TFLite models"""
    print("Testing TFLite models for Raspberry Pi 5...")
    
    models_dir = "models/"
    models_to_test = [
        "Lightweight-Face-Detection.tflite",
        "MobileFaceNet_9925_9680.tflite"
    ]
    
    all_tests_passed = True
    
    for model_name in models_to_test:
        model_path = os.path.join(models_dir, model_name)
        
        if os.path.exists(model_path):
            try:
                print(f"\nTesting {model_name}...")
                
                # Load model
                interpreter = tflite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                # Get model details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                print(f"  ✓ Model loaded successfully")
                print(f"  Input shape: {input_details[0]['shape']}")
                print(f"  Output shape: {output_details[0]['shape']}")
                
                # Test with dummy input
                if "Face-Detection" in model_name:
                    # Face detection model expects grayscale input
                    dummy_input = np.random.random((1, 480, 640, 1)).astype(np.float32)
                else:
                    # Face recognition model expects RGB input
                    dummy_input = np.random.random((1, 112, 112, 3)).astype(np.float32)
                
                interpreter.set_tensor(input_details[0]['index'], dummy_input)
                interpreter.invoke()
                
                # Get output
                output = interpreter.get_tensor(output_details[0]['index'])
                print(f"  Output shape: {output.shape}")
                print(f"  ✓ Inference test passed")
                
            except Exception as e:
                print(f"  ✗ Model test failed: {e}")
                all_tests_passed = False
        else:
            print(f"\n✗ Model not found: {model_path}")
            print("Please ensure the model files are in the models/ directory")
            all_tests_passed = False
    
    if all_tests_passed:
        print("\n✓ All model tests passed!")
        print("Models are ready for use with the face recognition system.")
    else:
        print("\n✗ Some model tests failed!")
        print("Please check the model files and try again.")
    
    return all_tests_passed

if __name__ == "__main__":
    success = test_models()
    exit(0 if success else 1) 