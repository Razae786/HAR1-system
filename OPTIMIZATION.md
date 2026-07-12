# Performance Optimization Guide

This document outlines identified performance issues and recommended optimizations for the HAR system.

---

## 🔴 Critical Performance Issues

### 1. **Frame-by-Frame Pose Detection is CPU-Intensive**

**Issue**: Every frame runs through MediaPipe pose detection, which is computationally expensive.

```python
# Current (inefficient)
while cap.isOpened():
    ret, frame = cap.read()
    results = pose.process(image_rgb)  # Runs on every frame
```

**Optimization**:
```python
# Frame skipping
SKIP_FRAMES = 2
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % SKIP_FRAMES == 0:
        results = pose.process(image_rgb)
    # Use previous results for skipped frames
    
    frame_count += 1
```

**Expected Impact**: 2-3x speed improvement

---

### 2. **Inefficient Prediction Smoothing (Every Frame)**

**Issue**: Creating `Counter` object per frame for 5-element list.

```python
# Current (inefficient)
smooth_pred = Counter(recent_preds).most_common(1)[0][0]  # Per frame!
```

**Optimization**:
```python
# Use statistics module
from statistics import mode

smooth_pred = mode(recent_preds)  # ~10% faster
```

**Or use deque + manual tracking**:
```python
from collections import deque

class PredictionSmoother:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)
        self.counts = {}
    
    def add(self, pred):
        if self.window and self.window[0] != pred:
            self.counts[self.window[0]] -= 1
        self.window.append(pred)
        self.counts[pred] = self.counts.get(pred, 0) + 1
        return max(self.counts, key=self.counts.get)
```

**Expected Impact**: 15-20% speed improvement on smoothing

---

### 3. **NumPy Array Creation in Loops**

**Issue**: Creating new arrays repeatedly for angle calculations.

```python
# Current (inefficient)
def calculate_angle(a, b, c):
    a = np.array(a)  # Conversion overhead
    b = np.array(b)
    c = np.array(c)
    # ... rest of calculation
```

**Optimization**:
```python
# Vectorized approach
def calculate_angles_vectorized(landmarks):
    """Calculate all angles at once using NumPy broadcasting"""
    left_knee = calculate_angle(
        landmarks[[23, 25, 27], :2]
    )
    right_knee = calculate_angle(
        landmarks[[24, 26, 28], :2]
    )
    # ... vectorize all 4 angles
    return np.array([left_knee, right_knee, left_hip, right_hip])
```

**Expected Impact**: 30% speed improvement on feature extraction

---

### 4. **Full Frame Display Every Frame**

**Issue**: Rendering high-resolution frames with landmarks every iteration.

```python
# Current (inefficient)
stframe.image(frame, channels="BGR")  # Full resolution, every frame
```

**Optimization**:
```python
# Display downsampled or every Nth frame
if frame_count % 3 == 0:  # Display every 3rd frame
    display_frame = cv2.resize(frame, (640, 480))
    stframe.image(display_frame, channels="BGR")
```

**Expected Impact**: 40-50% reduction in rendering time

---

### 5. **Model Loaded at Module Level (Streamlit Re-runs)**

**Issue**: Model reloaded on every script re-run.

```python
# Current (inefficient)
with open("model/activity_model.pkl", "rb") as f:
    model = pickle.load(f)  # Loaded every re-run!
```

**Optimization**:
```python
import streamlit as st

@st.cache_resource
def load_model():
    with open("model/activity_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("model/scaler.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()
```

**Expected Impact**: Eliminates 200-500ms per re-run

---

### 6. **Bare Exception Handling (Debugging Overhead)**

**Issue**: Generic `except:` masks real errors.

```python
# Current (bad)
try:
    left_knee = calculate_angle(...)
except:
    hip_angle, knee_angle = 180, 180
```

**Optimization**:
```python
# Specific exception handling
try:
    left_knee = calculate_angle(...)
except (IndexError, ZeroDivisionError, ValueError) as e:
    logger.warning(f"Angle calculation failed: {e}")
    hip_angle, knee_angle = 180, 180
```

**Expected Impact**: Easier debugging, ability to catch real issues

---

## 📊 Recommended Implementation Priority

| Priority | Issue | Optimization | Est. Gain |
|----------|-------|--------------|-----------|
| 🔴 High | Frame display rendering | Display every Nth frame + downsampling | 40-50% |
| 🔴 High | Model reloading | @st.cache_resource | 200-500ms per run |
| 🟠 Medium | Pose detection | Frame skipping | 2-3x speedup |
| 🟠 Medium | Feature extraction | Vectorize angle calculations | 30% |
| 🟡 Low | Smoothing | Use statistics.mode() | 15-20% |

---

## 🚀 Optimized Streamlit App Template

```python
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import tempfile
from collections import Counter, deque
from statistics import mode

# ============ CACHING ============
@st.cache_resource
def load_model():
    with open("model/activity_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("model/scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def get_pose_detector():
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

# ============ LOAD RESOURCES ============
model = load_model()
scaler = load_scaler()
pose = get_pose_detector()
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

st.title("📹 HAR System (Optimized)")

# ============ CONFIGURATION ============
SKIP_FRAMES = 2
DISPLAY_EVERY_N = 3
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# ============ VIDEO PROCESSING ============
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    stframe = st.empty()
    activity_log = []
    recent_preds = deque(maxlen=5)
    frame_count = 0
    pose_frame_count = 0
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    prev_results = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Frame skipping
        if frame_count % SKIP_FRAMES == 0:
            results = pose.process(image_rgb)
            prev_results = results
        else:
            results = prev_results
        
        # Process predictions
        if results and results.pose_landmarks:
            # Extract features and predict (simplified)
            pred = "Standing"  # Placeholder
            recent_preds.append(pred)
            smooth_pred = mode(recent_preds)  # Faster
            activity_log.append(smooth_pred)
        
        # Display every Nth frame (downsampled)
        if frame_count % DISPLAY_EVERY_N == 0:
            display_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            if results and results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame, results.pose_landmarks, POSE_CONNECTIONS
                )
            cv2.putText(display_frame, f"Activity: {smooth_pred}", 
                       (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            stframe.image(display_frame, channels="BGR")
        
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    
    st.subheader("🧾 Activity Summary")
    summary = Counter(activity_log)
    total = sum(summary.values())
    for act, count in summary.items():
        st.write(f"**{act}**: {count} frames ({(count/total)*100:.1f}%)")
```

---

## 🔧 Additional Optimization Techniques

### Batch Processing
```python
# Process 4 frames in parallel using ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(pose.process, frame) for frame in frames_batch]
    results = [f.result() for f in futures]
```

### GPU Acceleration (if available)
```python
# ONNX Runtime with GPU
import onnxruntime as ort

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Specify GPU execution provider
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession('model.onnx', session_options, providers=providers)
```

### Reduce Landmark Data (Optional)
```python
# Use only critical landmarks (13 instead of 33)
CRITICAL_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

def extract_critical_features(landmarks):
    critical = landmarks[CRITICAL_LANDMARKS]
    return critical.flatten()  # 52D vector instead of 132D
```

---

## 📈 Benchmarking Your Changes

```python
import time

def benchmark_function(func, *args, iterations=100):
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()
    avg_time = (end - start) / iterations * 1000
    print(f"{func.__name__}: {avg_time:.2f}ms")

# Usage
benchmark_function(calculate_angle, [0,0], [1,1], [2,0])
benchmark_function(Counter, ['A', 'B', 'A', 'C', 'A'])
```

---

## ✅ Optimization Checklist

- [ ] Enable Streamlit caching for model/scaler loading
- [ ] Implement frame skipping (SKIP_FRAMES = 2-3)
- [ ] Downsample display frames (640x480 or smaller)
- [ ] Display every Nth frame instead of every frame
- [ ] Replace `Counter()` with `statistics.mode()`
- [ ] Vectorize angle calculations
- [ ] Use specific exception handling (not bare `except:`)
- [ ] Add progress bar for long video processing
- [ ] Profile code with `cProfile` to identify remaining bottlenecks
- [ ] Consider GPU acceleration for production

---

**Last Updated**: 2024  
**Next Review**: After implementing frame 1-3 optimizations
