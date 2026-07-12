# Troubleshooting Guide

## Common Issues & Solutions

---

## Installation Issues

### ❌ "ModuleNotFoundError: No module named 'streamlit'"

**Cause**: Dependencies not installed or virtual environment not activated

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

---

### ❌ "Python command not found" (macOS/Linux)

**Cause**: Python not in PATH

**Solution**:
```bash
# Use python3 instead
python3 --version
python3 -m venv venv
source venv/bin/activate
```

---

### ❌ "pip: command not found"

**Cause**: pip not installed or not in PATH

**Solution**:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Runtime Issues

### ❌ "ImportError: No module named 'mediapipe'"

**Cause**: MediaPipe not installed correctly

**Solution**:
```bash
# Reinstall with verbose output
pip install mediapipe --upgrade --force-reinstall -v

# If still failing, try:
pip install mediapipe==0.10.0
```

---

### ❌ "streamlit: command not found"

**Cause**: Streamlit not in PATH

**Solution**:
```bash
# Run via Python module
python -m streamlit run streamlit_app.py
```

---

### ❌ "Address already in use" (Port 8501)

**Cause**: Port 8501 already being used

**Solution**:
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502
```

---

## Video Processing Issues

### ❌ "Cannot open video file"

**Cause**: Video codec not supported or file corrupted

**Solution**:
1. Try converting video to MP4: `ffmpeg -i input.avi output.mp4`
2. Check video format is supported (MP4, MOV, AVI, MKV)
3. Ensure file is not corrupted: `file video.mp4`

---

### ❌ "No Pose detected" on valid video

**Cause**: 
- Poor lighting
- Person too far from camera
- Multiple people in frame
- Extreme angles or occlusions

**Solution**:
1. Improve lighting conditions
2. Ensure person fills 30-70% of frame
3. Keep only one person in frame
4. Face camera directly (0-45° angle)
5. Minimize clothing occlusion

---

### ❌ "Empty video file" error

**Cause**: Video file is corrupted or empty

**Solution**:
```bash
# Check video properties
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of default=noprint_wrappers=1 video.mp4

# Repair video
ffmpeg -i corrupted.mp4 -c:v libx264 -c:a aac repaired.mp4
```

---

## Performance Issues

### ❌ "Very slow processing" or "Low FPS"

**Cause**: Full resolution processing, no frame skipping

**Solution**:
Edit `streamlit_app.py`:
```python
# Skip frames (process every 2nd frame)
SKIP_FRAMES = 2

# Reduce display resolution
TARGET_WIDTH = 480
TARGET_HEIGHT = 360

# Display every 3rd frame instead of every frame
if frame_count % 3 == 0:
    stframe.image(display_frame, channels="BGR")
```

See [OPTIMIZATION.md](OPTIMIZATION.md) for more details.

---

### ❌ "High CPU usage"

**Cause**: All frames processed, model loaded on every run

**Solution**:
```python
# Already implemented in streamlit_app.py:
@st.cache_resource
def load_model():
    return pickle.load(open("model/activity_model.pkl", "rb"))

# Implement frame skipping
if frame_count % 2 == 0:
    results = pose.process(image_rgb)
```

---

### ❌ "Out of Memory" on large videos

**Cause**: Loading entire video into memory

**Solution**:
1. Use frame-by-frame processing (already implemented)
2. Reduce frame resolution before processing
3. Split large video into smaller chunks

---

## Model Issues

### ❌ "FileNotFoundError: model/activity_model.pkl not found"

**Cause**: Model files missing

**Solution**:
```bash
# Check if model directory exists
ls model/

# If missing, train a new model
python app.py

# Verify files created
ls -la model/
```

---

### ❌ "Inaccurate predictions"

**Cause**: Model trained on different data distribution

**Solution**:
1. Collect videos with varied angles, lighting, clothing
2. Retrain model: `python app.py`
3. Verify training data has balanced classes
4. Check for data quality issues

---

### ❌ "Model predictions inconsistent"

**Cause**: No temporal smoothing or wrong smoothing window

**Solution**:
```python
# Increase smoothing window (more stable)
N = 7  # Instead of 5

# Or use moving average
recent_preds = deque(maxlen=N)
smooth_pred = Counter(recent_preds).most_common(1)[0][0]
```

---

## System-Specific Issues

### 🪟 Windows Issues

#### ❌ "ImportError: DLL load failed"

**Cause**: Missing Visual C++ redistributable

**Solution**:
1. Download Visual C++ redistributable from Microsoft
2. Install it
3. Restart terminal and Python

---

### 🍎 macOS Issues

#### ❌ "zsh: permission denied: streamlit"

**Cause**: Executable permission missing

**Solution**:
```bash
chmod +x venv/bin/streamlit
streamlit run streamlit_app.py
```

#### ❌ "dlopen failed" (MediaPipe on M1/M2)

**Cause**: Architecture incompatibility

**Solution**:
```bash
# Install Apple Silicon version
pip install --upgrade --force-reinstall mediapipe
```

---

### 🐧 Linux Issues

#### ❌ "ImportError: libGL.so.1 not found"

**Cause**: Missing OpenGL libraries

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx libsm6 libxext6

# Fedora/RHEL
sudo yum install mesa-libGL
```

---

## Debugging Tips

### Enable Verbose Logging

```python
# Add to streamlit_app.py
import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.info("Starting HAR app")
```

### Check Versions

```bash
python --version
pip list | grep -E "streamlit|mediapipe|opencv|scikit-learn"
```

### Profile Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
results = pose.process(image_rgb)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

---

## Getting Help

1. **Check Documentation**:
   - [README.md](README.md) - Overview
   - [SETUP.md](SETUP.md) - Installation
   - [OPTIMIZATION.md](OPTIMIZATION.md) - Performance
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Design

2. **Search Issues**: https://github.com/Razae786/HAR1-system/issues

3. **Report Bug**:
   - Include: OS, Python version, error message
   - Steps to reproduce
   - Expected vs actual behavior

4. **Ask Question**:
   - Use GitHub Discussions
   - Include minimal reproducible example

---

**Last Updated**: 2024
