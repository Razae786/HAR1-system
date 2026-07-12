# Human Activity Recognition (HAR) System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://mediapipe.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Real-time human activity recognition using pose estimation and machine learning. Detects **Standing**, **Walking**, **Sitting**, and **Falling** from video input with smooth prediction stabilization.

[Report Bug](https://github.com/Razae786/HAR1-system/issues) | [Request Feature](https://github.com/Razae786/HAR1-system/issues)

---

## 🚀 Features

- **Real-time Pose Detection**: Uses MediaPipe Holistic for 33-body landmark extraction
- **4 Activity Classes**: Standing, Walking, Sitting, Falling detection
- **Smooth Predictions**: Temporal smoothing with sliding window (reduces jitter)
- **Joint Angle Analysis**: Calculates hip flexion, knee flexion, and hand movement velocity
- **Streamlit Web Interface**: Upload video or use webcam for instant analysis
- **Lightweight Model**: Decision Tree classifier (~2MB) for fast inference
- **Visualization**: Real-time skeleton overlay with activity label and confidence

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Pose Estimation | MediaPipe Holistic |
| Feature Extraction | Custom joint angles + landmark coordinates |
| Classifier | scikit-learn Decision Tree |
| Frontend | Streamlit |
| Video Processing | OpenCV |
| Data Handling | NumPy, Pandas |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- Webcam (for live detection) or video file

### Installation

```bash
# Clone the repository
git clone https://github.com/Razae786/HAR1-system.git
cd HAR1-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
HAR1-system/
├── streamlit_app.py          # Main Streamlit application
├── app.py                    # Model training pipeline
├── test.csv                  # Test dataset
├── model/
│   ├── activity_model.pkl    # Pre-trained Decision Tree classifier
│   └── scaler.pkl            # Feature scaler
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── OPTIMIZATION.md
```

---

## 🔧 How It Works

1. **Pose Extraction**: MediaPipe extracts 33 body landmarks per frame
2. **Feature Engineering**: 
   - Normalized landmark coordinates (x, y, z, visibility)
   - Joint angles: hip flexion, knee flexion
   - Hand movement velocity (wrist displacement)
3. **Classification**: Decision Tree predicts activity from 132-dimensional feature vector
4. **Temporal Smoothing**: 5-frame sliding window majority vote stabilizes output

### Activity Detection Logic

```
Knee Angle < 100° → Sitting
Knee Angle > 160° AND Hip Angle > 150° → Standing
High Hand Movement + Upright Posture → Working
Otherwise → Model Prediction
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Inference Speed | ~30 FPS (CPU) |
| Model Size | 2.1 MB |
| Latency | <50ms per frame |
| Input Features | 132 (33 landmarks × 4 values) |

---

## 🎯 Use Cases

- **Elderly Care Monitoring**: Detect falls and anomalies in real-time
- **Fitness Tracking**: Count reps and classify exercise form
- **Workplace Safety**: Monitor ergonomic posture and activity patterns
- **Gait Analysis**: Rehabilitation progress tracking and biomechanics study

---

## 🚨 Known Performance Considerations

This project prioritizes ease of use over raw performance. For production deployments:

1. **Frame Skipping**: Process every Nth frame to reduce latency
2. **Resolution Scaling**: Downscale input frames before pose detection
3. **Model Caching**: Use `@st.cache_resource` to avoid reloading
4. **Batch Processing**: Process multiple frames simultaneously
5. **GPU Acceleration**: Deploy with CUDA/ONNX for faster inference

See [OPTIMIZATION.md](OPTIMIZATION.md) for detailed optimization strategies.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for pose estimation
- [Streamlit](https://streamlit.io/) for the awesome web framework
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [OpenCV](https://opencv.org/) for video processing

---

## 📧 Contact

**Razae786** - [GitHub](https://github.com/Razae786)

Project Link: [https://github.com/Razae786/HAR1-system](https://github.com/Razae786/HAR1-system)
