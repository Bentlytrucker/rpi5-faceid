# Troubleshooting Guide

## Common Issues and Solutions

### 1. Camera Issues

#### Camera Not Detected
**Symptoms**: Camera not working, "Camera initialization error"

**Solutions**:
```bash
# Check camera hardware
vcgencmd get_camera
# Should return: supported=1 detected=1

# Enable camera interface
sudo raspi-config
# Interface Options → Camera → Enable

# Check camera cable connection
# Ensure Camera Module 3 is properly connected
```

#### Camera Permission Errors
**Symptoms**: "Permission denied" when accessing camera

**Solutions**:
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again
# Or reboot: sudo reboot

# Check groups
groups $USER
# Should include 'video'
```

#### Camera Preview Not Showing
**Symptoms**: Camera works but no preview window

**Solutions**:
```bash
# Check if running in GUI environment
echo $DISPLAY

# If no display, enable VNC or use SSH with X11 forwarding
# For SSH: ssh -X pi@raspberrypi.local

# Test camera manually
libcamera-hello
```

### 2. Model Loading Issues

#### TFLite Runtime Import Error
**Symptoms**: "No module named 'tflite_runtime'"

**Solutions**:
```bash
# Check if virtual environment is activated
which python3
# Should show: /home/pi/venv/bin/python3

# Reinstall TFLite Runtime
pip install --force-reinstall --no-deps tflite-runtime

# Verify installation
python3 -c "import tflite_runtime; print('OK')"
```

#### Model Files Not Found
**Symptoms**: "Model not found" errors

**Solutions**:
```bash
# Check model files exist
ls -la models/

# Copy models from parent directory if needed
cp ../qualcomm_lightweight/Lightweight-Face-Detection.tflite models/
cp ../mobilefacenet_conversion/MobileFaceNet_9925_9680.tflite models/

# Check file permissions
chmod 644 models/*.tflite
```

#### Model Loading Errors
**Symptoms**: "Interpreter allocation failed"

**Solutions**:
```bash
# Check available memory
free -h

# Increase swap memory if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=100
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Restart system
sudo reboot
```

### 3. Performance Issues

#### Low FPS
**Symptoms**: FPS below 10, choppy video

**Solutions**:
```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase GPU memory
sudo raspi-config
# Performance Options → GPU Memory → 128

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon

# Check CPU temperature
vcgencmd measure_temp
# Should be below 80°C
```

#### High Memory Usage
**Symptoms**: System becomes slow, out of memory errors

**Solutions**:
```bash
# Check memory usage
htop

# Increase swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=200
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Close unnecessary applications
```

#### High CPU Usage
**Symptoms**: CPU usage above 80%

**Solutions**:
```bash
# Check what's using CPU
top

# Optimize camera settings in realtime_inference.py
# Reduce FRAME_WIDTH and FRAME_HEIGHT
# Increase FPS_TARGET

# Use lighter models if available
```

### 4. Face Recognition Issues

#### No Face Detection
**Symptoms**: No green boxes around faces

**Solutions**:
```bash
# Check confidence threshold
# In realtime_inference.py, try lowering CONFIDENCE_THRESHOLD
# Default: 0.15, try: 0.1

# Check lighting conditions
# Ensure good lighting for face detection

# Check camera focus
# Ensure camera is properly focused
```

#### False Positives
**Symptoms**: Detecting faces where there are none

**Solutions**:
```bash
# Increase confidence threshold
# In realtime_inference.py, increase CONFIDENCE_THRESHOLD
# Try: 0.2 or 0.25

# Check camera positioning
# Ensure camera is not pointing at reflective surfaces
```

#### Recognition Accuracy Issues
**Symptoms**: Wrong names or "Unknown" for known faces

**Solutions**:
```bash
# Adjust similarity threshold
# In realtime_inference.py, modify SIMILARITY_THRESHOLD
# Default: 0.6, try: 0.5 for more lenient, 0.7 for stricter

# Re-register faces with better lighting
# Ensure consistent lighting during registration

# Check face angle and distance
# Register faces from multiple angles
```

### 5. System Errors

#### Import Errors
**Symptoms**: "ModuleNotFoundError"

**Solutions**:
```bash
# Check virtual environment
source ~/venv/bin/activate

# Reinstall packages
pip install --force-reinstall --no-deps tflite-runtime opencv-python numpy psutil

# Check Python path
python3 -c "import sys; print(sys.path)"
```

#### Permission Errors
**Symptoms**: "Permission denied" for files or directories

**Solutions**:
```bash
# Fix file permissions
chmod 755 realtime_inference.py
chmod 644 models/*.tflite

# Fix directory permissions
chmod 755 models/
chmod 755 tests/

# Check ownership
ls -la
# Should be owned by pi user
```

#### Network Issues
**Symptoms**: Cannot download packages

**Solutions**:
```bash
# Check internet connection
ping 8.8.8.8

# Update package lists
sudo apt update

# Check DNS
nslookup google.com

# Try different DNS
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```

### 6. Debugging Tools

#### System Information
```bash
# Check system specs
cat /proc/cpuinfo | grep Model
cat /proc/meminfo | grep MemTotal
vcgencmd get_mem gpu

# Check camera
vcgencmd get_camera
ls /dev/video*

# Check Python environment
python3 --version
which python3
pip list
```

#### Performance Monitoring
```bash
# Real-time system monitoring
htop

# GPU memory usage
vcgencmd get_mem gpu

# CPU temperature
vcgencmd measure_temp

# Memory usage
free -h
```

#### Log Analysis
```bash
# Check system logs
dmesg | tail -20

# Check camera logs
journalctl -u camera.service

# Check Python errors
python3 realtime_inference.py 2>&1 | tee error.log
```

## Getting Help

### Before Asking for Help
1. Check this troubleshooting guide
2. Verify all prerequisites are met
3. Run diagnostic commands above
4. Check system logs

### Useful Commands for Diagnosis
```bash
# System info
uname -a
cat /etc/os-release
vcgencmd version

# Camera test
libcamera-hello -t 0

# Python test
python3 -c "import tflite_runtime, cv2, numpy, psutil; print('All imports OK')"

# Performance test
python3 tests/test_camera.py
python3 tests/test_models.py
```

### Reporting Issues
When reporting issues, include:
1. Raspberry Pi model and OS version
2. Error messages and logs
3. Steps to reproduce the issue
4. System specifications
5. What you've already tried 