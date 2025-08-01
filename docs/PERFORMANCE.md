# Performance Optimization Guide

## Performance Metrics

### Target Performance (Raspberry Pi 5)
- **FPS**: 10-15 FPS
- **Memory Usage**: ~200MB
- **CPU Usage**: 30-50%
- **Detection Accuracy**: 95%+
- **Recognition Accuracy**: 90%+

### Baseline Performance
- **FPS**: 5-8 FPS (unoptimized)
- **Memory Usage**: ~400MB
- **CPU Usage**: 60-80%

## Optimization Strategies

### 1. System-Level Optimizations

#### CPU Governor
```bash
# Set to performance mode for maximum CPU speed
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make permanent
echo 'GOVERNOR="performance"' | sudo tee -a /etc/default/cpufrequtils
```

#### GPU Memory Allocation
```bash
# Increase GPU memory for better performance
sudo raspi-config
# Performance Options → GPU Memory → 128MB

# Check current allocation
vcgencmd get_mem gpu
```

#### Disable Unnecessary Services
```bash
# Disable Bluetooth
sudo systemctl disable bluetooth
sudo systemctl stop bluetooth

# Disable WiFi power management
sudo iwconfig wlan0 power off

# Disable unnecessary daemons
sudo systemctl disable avahi-daemon
sudo systemctl disable triggerhappy
```

#### Memory Optimization
```bash
# Increase swap memory
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=200
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Check memory usage
free -h
```

### 2. Camera Optimizations

#### Frame Rate and Resolution
```python
# In realtime_inference.py, adjust these parameters:
FRAME_WIDTH = 640    # Try 480 for better performance
FRAME_HEIGHT = 480   # Try 360 for better performance
FPS_TARGET = 15      # Adjust based on performance

# Camera configuration
config = picam2.create_preview_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
    controls={"FrameDurationLimits": (33333, 33333)}  # 30 FPS max
)
```

#### Camera Buffer Optimization
```python
# Reduce buffer size for lower latency
picam2.configure(config)
picam2.set_controls({"FrameDurationLimits": (33333, 33333)})
picam2.start()
```

### 3. Model Optimizations

#### Model Quantization
- Use quantized models (already implemented)
- Models are optimized for Raspberry Pi
- Consider using smaller models if available

#### Inference Optimization
```python
# Pre-allocate tensors
interpreter.allocate_tensors()

# Use batch processing if possible
# Process multiple frames together

# Optimize input preprocessing
def preprocess_for_detection_pi(image):
    # Use INTER_LINEAR for better performance
    resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
    # ... rest of preprocessing
```

### 4. Code-Level Optimizations

#### Memory Management
```python
# Reuse arrays to reduce memory allocation
import numpy as np

# Pre-allocate arrays
input_buffer = np.zeros((1, 480, 640, 1), dtype=np.float32)
output_buffer = np.zeros((1, 60, 80, 1), dtype=np.float32)

# Use in-place operations
np.multiply(array, 255.0, out=array)
```

#### Loop Optimizations
```python
# Reduce function calls in main loop
def main_stream():
    # Pre-compute constants
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    
    while True:
        # Minimize function calls
        frame = capture_frame_picamera2(picam2)
        if frame is None:
            continue
            
        # Process frame
        # ...
```

#### Conditional Optimization
```python
# Only process face recognition when face is detected
if face_result:
    # Only then compute embeddings
    current_embedding = get_face_embedding_pi(models, face_roi)
    
    # Early exit for low confidence
    if confidence < 0.2:
        continue
```

### 5. Display Optimizations

#### Reduce UI Updates
```python
# Update system info less frequently
if current_time - last_system_check >= 5.0:  # Every 5 seconds
    system_info = get_system_info()
    last_system_check = current_time

# Reduce text rendering
cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```

#### Window Management
```python
# Use named window for better performance
cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)

# Reduce window updates
cv2.imshow('Face Recognition', frame)
cv2.waitKey(1)  # Minimal delay
```

### 6. Advanced Optimizations

#### Multi-threading (Optional)
```python
import threading
import queue

# Separate camera capture from processing
frame_queue = queue.Queue(maxsize=2)

def camera_thread():
    while True:
        frame = capture_frame_picamera2(picam2)
        if not frame_queue.full():
            frame_queue.put(frame)

def processing_thread():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Process frame
```

#### Model Caching
```python
# Cache model outputs for similar inputs
import functools

@functools.lru_cache(maxsize=100)
def cached_face_embedding(face_hash):
    # Compute embedding only for new faces
    return get_face_embedding_pi(models, face_img)
```

## Performance Monitoring

### Real-time Monitoring
```python
import psutil
import time

def monitor_performance():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"CPU: {cpu_percent:.1f}%")
    print(f"Memory: {memory.percent:.1f}%")
    print(f"FPS: {current_fps:.1f}")
```

### Performance Logging
```python
import logging

logging.basicConfig(filename='performance.log', level=logging.INFO)

def log_performance():
    logging.info(f"FPS: {current_fps}, CPU: {cpu_percent}, Memory: {memory_percent}")
```

## Benchmarking

### Performance Test Script
```python
# tests/benchmark.py
import time
import psutil

def benchmark_performance():
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 60:  # 1 minute test
        # Process frame
        frame_count += 1
    
    avg_fps = frame_count / 60
    print(f"Average FPS: {avg_fps:.1f}")
```

### Memory Profiling
```python
import tracemalloc

tracemalloc.start()

# Run your code here

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

## Optimization Checklist

### System Level
- [ ] CPU governor set to performance
- [ ] GPU memory increased to 128MB
- [ ] Unnecessary services disabled
- [ ] Swap memory increased
- [ ] System temperature below 80°C

### Application Level
- [ ] Frame resolution optimized
- [ ] Frame rate adjusted
- [ ] Model quantization enabled
- [ ] Memory allocations minimized
- [ ] UI updates reduced

### Code Level
- [ ] Function calls minimized in main loop
- [ ] Arrays pre-allocated
- [ ] Early exits implemented
- [ ] Conditional processing optimized
- [ ] Display updates optimized

## Expected Performance Gains

### After System Optimizations
- **FPS**: +20-30%
- **Memory Usage**: -10-15%
- **CPU Usage**: -15-20%

### After Code Optimizations
- **FPS**: +15-25%
- **Memory Usage**: -20-30%
- **CPU Usage**: -10-15%

### After Combined Optimizations
- **FPS**: +40-60%
- **Memory Usage**: -30-40%
- **CPU Usage**: -25-35%

## Troubleshooting Performance Issues

### Low FPS
1. Check CPU temperature
2. Reduce frame resolution
3. Increase GPU memory
4. Disable unnecessary services

### High Memory Usage
1. Increase swap memory
2. Reduce frame buffer size
3. Implement memory pooling
4. Use lighter models

### High CPU Usage
1. Set CPU governor to performance
2. Optimize main loop
3. Reduce function calls
4. Use conditional processing 