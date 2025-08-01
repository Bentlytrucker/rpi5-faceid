#!/bin/bash

echo "=== Raspberry Pi 5 Face Recognition System Installation ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    print_warning "This script is designed for Raspberry Pi. Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y \
    python3-picamera2 \
    python3-pip \
    python3-venv \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjasper-dev \
    libqtcore4 \
    libqtgui4 \
    libqt4-test \
    python3-dev

# Create virtual environment
print_status "Creating virtual environment..."
python3 -m venv ~/venv --system-site-packages

# Activate virtual environment
print_status "Activating virtual environment..."
source ~/venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --no-deps tflite-runtime
pip install --no-deps opencv-python
pip install numpy==1.24.2
pip install psutil

# Create models directory
print_status "Creating models directory..."
mkdir -p models

# Copy model files if they exist in parent directory
if [ -f "../qualcomm_lightweight/Lightweight-Face-Detection.tflite" ]; then
    print_status "Copying face detection model..."
    cp ../qualcomm_lightweight/Lightweight-Face-Detection.tflite models/
fi

if [ -f "../mobilefacenet_conversion/MobileFaceNet_9925_9680.tflite" ]; then
    print_status "Copying face recognition model..."
    cp ../mobilefacenet_conversion/MobileFaceNet_9925_9680.tflite models/
fi

# Create test files
print_status "Creating test files..."

# Camera test
cat > tests/test_camera.py << 'EOF'
import cv2
import numpy as np
from picamera2 import Picamera2

print("Testing Picamera2...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

for i in range(30):
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Picamera2 Test', frame_bgr)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()
print("✓ Picamera2 test completed")
EOF

# Model test
cat > tests/test_models.py << 'EOF'
import tflite_runtime.interpreter as tflite
import numpy as np
import os

def test_models():
    models_dir = "models/"
    
    # Test face detection model
    detection_path = os.path.join(models_dir, "Lightweight-Face-Detection.tflite")
    if os.path.exists(detection_path):
        try:
            interpreter = tflite.Interpreter(model_path=detection_path)
            interpreter.allocate_tensors()
            print("✓ Face detection model loaded successfully")
        except Exception as e:
            print(f"✗ Face detection model error: {e}")
    else:
        print("✗ Face detection model not found")
    
    # Test face recognition model
    recognition_path = os.path.join(models_dir, "MobileFaceNet_9925_9680.tflite")
    if os.path.exists(recognition_path):
        try:
            interpreter = tflite.Interpreter(model_path=recognition_path)
            interpreter.allocate_tensors()
            print("✓ Face recognition model loaded successfully")
        except Exception as e:
            print(f"✗ Face recognition model error: {e}")
    else:
        print("✗ Face recognition model not found")

if __name__ == "__main__":
    test_models()
EOF

# Make test files executable
chmod +x tests/test_camera.py
chmod +x tests/test_models.py

print_status "Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Enable camera interface: sudo raspi-config"
echo "2. Set GPU memory to 128MB: sudo raspi-config"
echo "3. Reboot: sudo reboot"
echo "4. Activate virtual environment: source ~/venv/bin/activate"
echo "5. Test camera: python3 tests/test_camera.py"
echo "6. Test models: python3 tests/test_models.py"
echo "7. Run face recognition: python3 realtime_inference.py"
echo ""
echo "For detailed instructions, see README.md" 