#!/usr/bin/env python3
"""
Camera Test Script for Raspberry Pi 5 Face Recognition System

This script tests the Raspberry Pi Camera Module 3 functionality
using Picamera2 library.

Usage:
    python3 tests/test_camera.py
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time

def test_camera():
    """Test camera functionality"""
    print("Testing Raspberry Pi Camera Module 3...")
    
    try:
        # Initialize camera
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        print("✓ Camera initialized successfully")
        print("Capturing 30 frames for testing...")
        
        # Capture and display frames
        for i in range(30):
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add frame counter
            cv2.putText(frame_bgr, f"Frame: {i+1}/30", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Test - Raspberry Pi Camera Module 3', frame_bgr)
            
            # Press 'q' to quit early
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        picam2.close()
        cv2.destroyAllWindows()
        print("✓ Camera test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_camera()
    exit(0 if success else 1) 