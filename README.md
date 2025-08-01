# Raspberry Pi 5 Face Recognition

Real-time face recognition system optimized for Raspberry Pi 5 with Camera Module 3 using TensorFlow Lite Runtime.

## Features

- **Real-time Face Detection**: Lightweight face detection using Qualcomm's optimized model
- **Face Recognition**: MobileFaceNet for face embedding and comparison
- **Camera Integration**: Picamera2 for optimal Raspberry Pi Camera Module 3 performance
- **Memory Optimized**: TFLite Runtime for reduced memory usage
- **System Monitoring**: Real-time CPU and RAM usage display
- **Face Database**: Save and load registered faces
- **Raspberry Pi 5 Optimized**: Specifically designed for Pi 5's ARM Cortex-A76 cores

## Hardware Requirements

- Raspberry Pi 5 (4GB RAM recommended)
- Raspberry Pi Camera Module 3
- MicroSD card (32GB+ recommended)
- Power supply (5V/3A recommended)

## Software Requirements

- Raspberry Pi OS (Bookworm) or newer
- Python 3.7+
- Virtual environment (recommended)

## Installation

```bash
pip install -r requirements.txt
```

## Installation Guide

### Manual Installation

#### 1. System Dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-picamera2 -y
```

#### 2. Create Virtual Environment
```bash
python3 -m venv ~/venv --system-site-packages
source ~/venv/bin/activate
```

#### 3. Install Python Dependencies
```bash
pip install --no-deps tflite-runtime
pip install --no-deps opencv-python
pip install numpy==1.24.2
pip install psutil
```

#### 4. Enable Camera Interface
```bash
sudo raspi-config
# Interface Options → Camera → Enable
```

#### 5. Reboot System
```bash
sudo reboot
```

## Usage

### Controls
- **'r'**: Register detected face
- **'s'**: Save face database
- **'l'**: Load face database
- **'q'**: Quit application

### Face Registration
1. Position face in camera view
2. Press 'r' key
3. Enter name when prompted
4. Face is registered for future recognition

### Performance Monitoring
- Real-time FPS display
- CPU usage percentage
- RAM usage percentage
- Number of registered faces

## Model Information

### Face Detection Model
- **Model**: Lightweight Face Detection (Qualcomm optimized)
- **Source**: [Qualcomm AI Hub](https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/face_det_lite)
- **Input**: 640x480 grayscale image
- **Output**: Bounding boxes and confidence scores
- **Optimization**: Quantized for Raspberry Pi
- **License**: Please check the original repository for license information

### Face Recognition Model
- **Model**: MobileFaceNet
- **Source**: [Sirius AI MobileFaceNet TF](https://github.com/sirius-ai/MobileFaceNet_TF/tree/master)
- **Input**: 112x112 RGB face image
- **Output**: 192-dimensional face embedding
- **Accuracy**: 99.25% on LFW dataset
- **License**: Please check the original repository for license information

## Model Conversion Process

### Converting MobileFaceNet from .pb to .tflite

The MobileFaceNet model was converted from TensorFlow protobuf (.pb) format to TensorFlow Lite (.tflite) format for optimized inference on Raspberry Pi 5.

#### Conversion Steps:

1. **Download the original .pb model**
   ```bash
   # Download MobileFaceNet_9925_9680.pb from the original repository
   wget <model_url>/MobileFaceNet_9925_9680.pb
   ```

2. **Install TensorFlow (for conversion)**
   ```bash
   pip install tensorflow==2.14.0
   ```

3. **Convert using TensorFlow Lite Converter**
   ```python
   import tensorflow as tf
   
   # Load the .pb model
   converter = tf.lite.TFLiteConverter.from_saved_model('MobileFaceNet_9925_9680.pb')
   
   # Set optimization flags
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.target_spec.supported_types = [tf.float16]
   
   # Convert to TFLite
   tflite_model = converter.convert()
   
   # Save the converted model
   with open('MobileFaceNet_9925_9680.tflite', 'wb') as f:
       f.write(tflite_model)
   ```

4. **Verify conversion**
   ```python
   import tflite_runtime.interpreter as tflite
   
   # Test the converted model
   interpreter = tflite.Interpreter(model_path='MobileFaceNet_9925_9680.tflite')
   interpreter.allocate_tensors()
   print("✓ Model converted successfully")
   ```



## Performance

### Typical Performance (Raspberry Pi 5 + Camera Module 3)
- **FPS**: 10-15 FPS
- **Memory Usage**: ~200MB
- **CPU Usage**: 30-50%
- **Detection Accuracy**: 95%+
- **Camera Resolution**: 640x480 (optimized for performance)
- **Model Size**: ~5MB (converted from .pb format)


### Customizing Parameters
Edit constants in `realtime_inference.py`:
- `CONFIDENCE_THRESHOLD`: Face detection confidence (default: 0.15)
- `SIMILARITY_THRESHOLD`: Face recognition threshold (default: 0.6)
- `FRAME_WIDTH/HEIGHT`: Camera resolution (default: 640x480)


## Model Sources

### Face Detection Model
- **Source**: [Qualcomm AI Hub - Face Detection Lite](https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/face_det_lite)
- **Model**: Lightweight Face Detection (Qualcomm optimized)
- **License**: Please check the original repository for license information
- **Optimization**: Quantized for Raspberry Pi

### Face Recognition Model
- **Source**: [MobileFaceNet TensorFlow Implementation](https://github.com/sirius-ai/MobileFaceNet_TF/tree/master)
- **Model**: MobileFaceNet
- **License**: Please check the original repository for license information
- **Accuracy**: 99.25% on LFW dataset

## Acknowledgments

- [Qualcomm AI Hub](https://github.com/quic/ai-hub-models) for lightweight face detection model
- [Sirius AI](https://github.com/sirius-ai/MobileFaceNet_TF) for MobileFaceNet TensorFlow implementation
- Raspberry Pi Foundation for Raspberry Pi 5 and Camera Module 3
- TensorFlow Lite team for optimized runtime
- ARM for Cortex-A76 architecture optimization 
