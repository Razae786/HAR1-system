# Quick Start Guide (5 Minutes)

## ⚡ Get Running in 5 Steps

### Step 1: Install Python (1 min)
```bash
# Check if you have Python 3.9+
python --version

# If not, download from https://www.python.org/downloads/
```

### Step 2: Clone & Setup (1 min)
```bash
git clone https://github.com/Razae786/HAR1-system.git
cd HAR1-system

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate        # macOS/Linux
# OR
venv\Scripts\activate           # Windows
```

### Step 3: Install Dependencies (2 min)
```bash
pip install -r requirements.txt
```

### Step 4: Run Application (1 min)
```bash
streamlit run streamlit_app.py
```

### Step 5: Upload Video
1. Open http://localhost:8501
2. Click "Upload a video file"
3. Select your video (MP4, MOV, AVI, MKV)
4. Watch predictions!

---

## 🎥 What to Expect

✅ Real-time skeleton overlay on video
✅ Activity predictions (Standing, Walking, Sitting, Falling)
✅ Activity summary at bottom
✅ Frame-by-frame analysis

---

## 📱 Test Videos

Try with:
- Person standing still (2-3 seconds)
- Person walking (5-10 seconds)
- Person sitting down (3-5 seconds)
- Person lying down (falling simulation)

**Best Results**:
- Good lighting
- Full body visible
- Clear background
- Video at least 2 seconds long

---

## 🚀 Next Steps

- [Read Full Documentation](README.md)
- [Optimize Performance](OPTIMIZATION.md)
- [Contribute](CONTRIBUTING.md)
- [Understand Architecture](ARCHITECTURE.md)

---

## ❓ Issues?

Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue.
