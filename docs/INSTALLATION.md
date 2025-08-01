# Installation Guide

## Prerequisites

### Hardware Requirements
- Raspberry Pi 5 (4GB RAM recommended)
- Camera Module 3
- MicroSD card (32GB+ recommended)
- Power supply (5V/3A recommended)

### Software Requirements
- Raspberry Pi OS (Bookworm) or newer
- Python 3.7+
- Internet connection for package installation

## Step-by-Step Installation

### 1. System Preparation

#### Update Raspberry Pi OS
```bash
sudo apt update && sudo apt upgrade -y
```

#### Enable Camera Interface
```bash
sudo raspi-config
```
Navigate to:
- **Interface Options** → **Camera** → **Enable**
- **Performance Options** → **GPU Memory** → **128**

#### Reboot System
```bash
sudo reboot
```

### 2. Install System Dependencies

#### Install Picamera2 and Python Tools
```bash
sudo apt install -y python3-picamera2 python3-pip python3-venv
```

#### Install Additional System Libraries
```bash
sudo apt install -y \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjasper-dev \
    libqtcore4 \
    libqtgui4 \
    libqt4-test \
    python3-dev
```

### 3. Create Virtual Environment

#### Create Environment with System Site Packages
```bash
python3 -m venv ~/venv --system-site-packages
```

#### Activate Environment
```bash
source ~/venv/bin/activate
```

### 4. Install Python Dependencies

#### Install TFLite Runtime (Lightweight)
```bash
pip install --no-deps tflite-runtime
```

#### Install OpenCV
```bash
pip install --no-deps opencv-python
```

#### Install Other Dependencies
```bash
pip install numpy==1.24.2
pip install psutil
```

### 5. Verify Installation

#### Test Camera
```bash
python3 tests/test_camera.py
```

#### Test Models
```bash
python3 tests/test_models.py
```

## Automated Installation

### Using Install Script
```bash
chmod +x install.sh
./install.sh
```

## Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check camera status
vcgencmd get_camera

# Should return: supported=1 detected=1
```

#### Permission Errors
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again, or reboot
```

#### Memory Issues
```bash
# Check available memory
free -h

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=100
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### Import Errors
```bash
# Check if virtual environment is activated
which python3

# Should show: /home/pi/venv/bin/python3

# Reinstall packages if needed
pip install --force-reinstall --no-deps tflite-runtime
```

### Performance Optimization

#### GPU Memory
- Set to 128MB in raspi-config
- Restart after changes

#### CPU Governor
```bash
# Set to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

#### Disable Unnecessary Services
```bash
# Disable Bluetooth if not needed
sudo systemctl disable bluetooth

# Disable WiFi power management
sudo iwconfig wlan0 power off
```

## Verification Checklist

- [ ] Camera interface enabled
- [ ] GPU memory set to 128MB
- [ ] Virtual environment created and activated
- [ ] TFLite Runtime installed
- [ ] OpenCV installed
- [ ] Camera test passes
- [ ] Model test passes
- [ ] System has sufficient memory

## Next Steps

After successful installation:

1. **Test the system**: `python3 realtime_inference.py`
2. **Register faces**: Press 'r' when face is detected
3. **Save database**: Press 's' to save registered faces
4. **Monitor performance**: Check FPS and system usage

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Check system logs: `dmesg | tail -20`
4. Test camera manually: `libcamera-hello` 