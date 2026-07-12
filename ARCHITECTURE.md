# Architecture Overview

## System Design

The HAR1-System is a real-time human activity recognition pipeline built on MediaPipe pose estimation and scikit-learn classification.

```
┌─────────────────────────────────────────────────────────┐
│                  VIDEO INPUT                            │
│              (MP4, MOV, AVI, MKV)                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  VIDEO FRAME EXTRACTION    │
        │   (OpenCV VideoCapture)    │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │  POSE LANDMARK EXTRACTION              │
        │  (MediaPipe - 33 landmarks per frame)  │
        └────────────┬───────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │  FEATURE ENGINEERING                   │
        │  - Coordinate normalization (x,y,z)    │
        │  - Joint angle calculation             │
        │  - Hand movement velocity              │
        │  - Visibility confidence               │
        │  Result: 132-dimensional feature vector│
        └────────────┬───────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │  FEATURE SCALING                       │
        │  (StandardScaler normalization)        │
        └────────────┬───────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │  ACTIVITY CLASSIFICATION               │
        │  (Decision Tree Classifier)            │
        │  Output: Activity Prediction           │
        └────────────┬───────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │  TEMPORAL SMOOTHING                    │
        │  (5-frame sliding window majority vote)│
        │  Output: Smoothed Activity Label       │
        └────────────┬───────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │  VISUALIZATION & OUTPUT                │
        │  - Pose skeleton overlay               │
        │  - Activity label display              │
        │  - Activity summary statistics         │
        └────────────────────────────────────────┘
```

---

## Component Details

### 1. Pose Extraction (MediaPipe)
- Detects 33 body landmarks (pose landmarks)
- Each landmark has: x, y, z coordinates + visibility confidence
- Real-time on CPU (~10-15ms per frame)
- Robust to occlusions and multiple people

### 2. Feature Engineering
**Input**: 33 landmarks × 4 values = 132 raw features

**Extracted Features**:
- Hip angle: Angle between left/right hip and knee
- Knee angle: Angle between hip and ankle
- Hand movement: L2 norm of hand displacement
- Confidence: Visibility scores

### 3. Classification
- **Model**: Decision Tree (scikit-learn)
- **Training**: Trained on annotated pose data
- **Classes**: Standing, Walking, Sitting, Falling
- **Inference**: <5ms per sample on CPU

### 4. Smoothing
- Window size: 5 frames
- Method: Majority voting
- Purpose: Reduce jitter and false positives

---

## File Structure

```
HAR1-system/
│
├── streamlit_app.py           # Main application (inference)
│   ├── load_model()           # Load pre-trained classifier
│   ├── extract_pose_features() # Feature extraction
│   ├── calculate_angle()      # Angle computation
│   └── Video processing loop  # Main inference pipeline
│
├── app.py                     # Training script
│   ├── Load training data
│   ├── Preprocessing
│   ├── Model training
│   └── Model serialization
│
├── model/
│   ├── activity_model.pkl     # Trained Decision Tree
│   └── scaler.pkl             # StandardScaler fitted on train data
│
├── requirements.txt           # Python dependencies
├── README.md                  # User documentation
├── CONTRIBUTING.md            # Contribution guidelines
├── OPTIMIZATION.md            # Performance optimization guide
├── ARCHITECTURE.md            # This file
└── LICENSE                    # MIT License
```

---

## Data Flow

### Training Phase (app.py)
```
train.csv
   ↓
[Load & Validate]
   ↓
[Extract features from landmarks]
   ↓
[StandardScaler.fit_transform()]
   ↓
[Train-test split (80-20)]
   ↓
[DecisionTreeClassifier.fit()]
   ↓
[Evaluation metrics]
   ↓
[Serialize model & scaler → pickle]
```

### Inference Phase (streamlit_app.py)
```
Video Upload
   ↓
[Frame extraction]
   ↓
[MediaPipe pose detection]
   ↓
[Feature extraction]
   ↓
[Scaler.transform()]
   ↓
[Model.predict()]
   ↓
[Temporal smoothing]
   ↓
[Display visualization]
   ↓
Activity Summary
```

---

## Key Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.1 | Web UI framework |
| mediapipe | 0.10.0 | Pose estimation |
| opencv-python | 4.8.0 | Video processing |
| scikit-learn | 1.3.0 | ML model & scaling |
| numpy | 1.24.3 | Numerical computing |
| pandas | 2.0.3 | Data manipulation |

---

## Performance Characteristics

### Per-Frame Latency (CPU)
- MediaPipe pose detection: 15-25ms
- Feature extraction: 2-3ms
- Model inference: 1-2ms
- Visualization: 5-10ms
- **Total**: ~25-40ms per frame (25-40 FPS achievable)

### Memory Usage
- Model size: 2.1 MB
- Scaler size: <100 KB
- Per-video buffering: ~50MB (for 1min @ 1080p)

### Scalability
- Single video stream: ✅ Supported
- Multiple streams: ⚠️ Requires optimization (see OPTIMIZATION.md)
- GPU support: ❌ Not implemented (but possible with ONNX)

---

## Future Improvements

- [ ] GPU acceleration with ONNX Runtime
- [ ] Batch processing for multiple videos
- [ ] Additional activity classes
- [ ] Skeleton keypoint heatmaps
- [ ] Ensemble modeling (multiple classifiers)
- [ ] Real-time webcam support
- [ ] Activity duration statistics
- [ ] Confidence score visualization

---

## Troubleshooting

### Issue: "No Pose" predictions on valid video
- **Cause**: Lighting, occlusion, or poor pose visibility
- **Fix**: Ensure good lighting and full body visibility

### Issue: Slow inference
- **Cause**: Full HD video processing
- **Fix**: Enable frame skipping (see OPTIMIZATION.md)

### Issue: Inaccurate predictions
- **Cause**: Model trained on different pose distribution
- **Fix**: Retrain model on your dataset

---

**Architecture Last Updated**: 2024  
**Version**: 1.0
