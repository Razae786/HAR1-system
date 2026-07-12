# Setup Guide for HAR1-System

## 🖥️ System Requirements

### Minimum Requirements
- Python 3.9 or higher
- 4GB RAM
- 2GB disk space
- Modern CPU (Intel i5/AMD Ryzen 5 or better)
- Webcam or video files for input

### Recommended
- Python 3.10+
- 8GB+ RAM
- SSD for faster video processing
- NVIDIA GPU (optional, for faster inference)

### Supported Operating Systems
- ✅ Windows 10/11
- ✅ macOS 10.13+
- ✅ Ubuntu 18.04+
- ✅ Other Linux distributions

---

## 📦 Installation Steps

### Step 1: Install Python

**Windows:**
1. Download from https://www.python.org/downloads/
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Complete installation

**macOS:**
```bash
brew install python@3.9
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install python3.9 python3.9-venv python3-pip
```

### Step 2: Clone Repository

```bash
git clone https://github.com/Razae786/HAR1-system.git
cd HAR1-system
```

### Step 3: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal.

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Expected output:
```
Successfully installed streamlit-1.28.1 opencv-python-4.8.0.76 mediapipe-0.10.0 ...
```

### Step 5: Download Pre-trained Model

The model files should be in `model/` directory:
- `activity_model.pkl` (Decision Tree classifier)
- `scaler.pkl` (Feature scaler)

If missing, contact the repo maintainer or retrain using `app.py`.

### Step 6: Verify Installation

```bash
python -c "import streamlit; import mediapipe; import cv2; print('✅ All dependencies installed!')"
```

---

## 🚀 Running the Application

### Start Streamlit App

```bash
streamlit run streamlit_app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

### Open in Browser

Navigate to: **http://localhost:8501**

You should see:
- Title: "📹 Human Activity Recognition (Pose-Based)"
- Upload button for video files
- Real-time activity display area

---

## 🎬 First Steps

### Option 1: Upload Your Own Video

1. Click **"Upload a video file"**
2. Select a video (MP4, MOV, AVI, MKV)
3. Wait for processing
4. View skeleton overlay and activity predictions
5. See activity summary at bottom

### Option 2: Test with Sample Video

If no model files are present:

```bash
# Train a new model
python app.py

# Then run the app
streamlit run streamlit_app.py
```

---

## 🔧 Troubleshooting Installation

### Issue: Python command not found
**Solution:**
```bash
# Try python3 instead
python3 --version
python3 -m venv venv
```

### Issue: Permission denied on macOS/Linux
**Solution:**
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

### Issue: MediaPipe fails to install
**Solution:**
```bash
pip install --upgrade pip setuptools wheel
pip install mediapipe --no-cache-dir
```

### Issue: OpenCV import error
**Solution:**
```bash
pip install opencv-python --upgrade --force-reinstall
```

---

## 🌐 Deploying to Streamlit Cloud

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### Step 2: Connect to Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Click "New app"
3. Select your GitHub repo
4. Select `streamlit_app.py` as main file
5. Deploy!

### Step 3: Share Live Demo
Your app will be live at: `https://your-username-HAR1-system.streamlit.app`

---

## 📝 Configuration

### Adjust Performance Settings

Edit `streamlit_app.py` to optimize for your hardware:

```python
# Frame skipping (increase for faster processing)
SKIP_FRAMES = 2  # Process every 2nd frame

# Display resolution
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# Prediction smoothing window
N = 5  # Number of frames to average
```

### Model Confidence Thresholds

```python
# Lower = more detections, higher = stricter
min_detection_confidence = 0.7
min_tracking_confidence = 0.7
```

---

## ✅ Verification Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Model files present in `model/`
- [ ] Streamlit app runs without errors
- [ ] Web interface accessible at localhost:8501
- [ ] Sample video processes successfully

---

## 📞 Need Help?

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Review [QUICK_START.md](QUICK_START.md)
- Open an [GitHub Issue](https://github.com/Razae786/HAR1-system/issues)

---

**Setup Last Updated**: 2024
