#!/usr/bin/env python3
"""
Model Conversion Script for Raspberry Pi 5 Face Recognition System

This script converts MobileFaceNet from TensorFlow protobuf (.pb) format
to TensorFlow Lite (.tflite) format for optimized inference on Raspberry Pi 5.

Requirements:
- TensorFlow 2.14.0 (for conversion)
- Original MobileFaceNet_9925_9680.pb file

Usage:
    python3 convert_model.py
"""

import tensorflow as tf
import os
import sys

def convert_pb_to_tflite(pb_path, tflite_path):
    """
    Convert TensorFlow protobuf model to TensorFlow Lite format
    
    Args:
        pb_path (str): Path to the .pb model file
        tflite_path (str): Output path for the .tflite model
    """
    try:
        print(f"Converting {pb_path} to {tflite_path}...")
        
        # Load the .pb model
        converter = tf.lite.TFLiteConverter.from_saved_model(pb_path)
        
        # Set optimization flags for Raspberry Pi 5
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Enable experimental features for better optimization
        converter.experimental_new_converter = True
        
        # Convert to TFLite
        tflite_model = converter.convert()
        
        # Save the converted model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get file sizes for comparison
        pb_size = os.path.getsize(pb_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
        
        print(f"✓ Conversion completed successfully!")
        print(f"  Original .pb size: {pb_size:.2f} MB")
        print(f"  Converted .tflite size: {tflite_size:.2f} MB")
        print(f"  Size reduction: {((pb_size - tflite_size) / pb_size * 100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False

def verify_tflite_model(tflite_path):
    """
    Verify the converted TFLite model can be loaded
    
    Args:
        tflite_path (str): Path to the .tflite model file
    """
    try:
        # Test loading the model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✓ Model verification successful!")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False

def main():
    """Main conversion function"""
    print("=== MobileFaceNet Model Conversion for Raspberry Pi 5 ===")
    
    # File paths
    pb_model_path = "MobileFaceNet_9925_9680.pb"
    tflite_model_path = "models/MobileFaceNet_9925_9680.tflite"
    
    # Check if .pb file exists
    if not os.path.exists(pb_model_path):
        print(f"✗ Error: {pb_model_path} not found!")
        print("Please download the MobileFaceNet_9925_9680.pb file first.")
        print("You can find it in the original repository:")
        print("https://github.com/sirius-ai/MobileFaceNet_TF")
        return False
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Convert the model
    if convert_pb_to_tflite(pb_model_path, tflite_model_path):
        # Verify the converted model
        if verify_tflite_model(tflite_model_path):
            print("\n✓ Model conversion and verification completed successfully!")
            print(f"Converted model saved to: {tflite_model_path}")
            print("\nYou can now use this model with the Raspberry Pi 5 face recognition system.")
            return True
        else:
            print("\n✗ Model verification failed!")
            return False
    else:
        print("\n✗ Model conversion failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 