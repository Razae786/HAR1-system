import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import tempfile
from collections import Counter

# Load trained model and scaler
with open("model/activity_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# MediaPipe Setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

st.title("📹 Human Activity Recognition (Pose-Based)")

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_pose_features(results):
    if not results.pose_landmarks:
        # Return 5 default values so unpacking never fails
        return None, 0, 0, [0, 0], [0, 0]

    landmarks = results.pose_landmarks.landmark
    pose_row = []
    for lm in landmarks:
        pose_row.extend([lm.x, lm.y, lm.z, lm.visibility])
    pose_array = np.array(pose_row[:132]).reshape(1, -1)

    def get_coords(idx):
        lm = landmarks[idx]
        return [lm.x, lm.y]

    try:
        left_knee = calculate_angle(get_coords(23), get_coords(25), get_coords(27))
        right_knee = calculate_angle(get_coords(24), get_coords(26), get_coords(28))
        left_hip = calculate_angle(get_coords(11), get_coords(23), get_coords(25))
        right_hip = calculate_angle(get_coords(12), get_coords(24), get_coords(26))
        hip_angle = (left_hip + right_hip) / 2
        knee_angle = (left_knee + right_knee) / 2

        left_hand = get_coords(15)
        right_hand = get_coords(16)
    except:
        hip_angle, knee_angle = 180, 180
        left_hand, right_hand = [0, 0], [0, 0]

    return pose_array, hip_angle, knee_angle, left_hand, right_hand

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    activity_log = []

    recent_preds = []
    N = 5

    prev_left_hand = prev_right_hand = None

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            features, hip_angle, knee_angle, left_hand, right_hand = extract_pose_features(results)
            if features is not None:
                features_scaled = scaler.transform(features)
                pred = model.predict(features_scaled)[0]

                # Posture-based adjustments
                if knee_angle < 100:
                    pred = "Sitting"
                elif knee_angle > 160 and hip_angle > 150:
                    pred = "Standing"

                # Detect repetitive hand movement = Working
                if prev_left_hand and prev_right_hand:
                    lh_move = np.linalg.norm(np.array(left_hand) - np.array(prev_left_hand))
                    rh_move = np.linalg.norm(np.array(right_hand) - np.array(prev_right_hand))
                    if (lh_move + rh_move) > 0.05 and knee_angle > 90 and hip_angle > 90:
                        pred = "Working"

                prev_left_hand = left_hand
                prev_right_hand = right_hand
            else:
                pred = "No Pose"

            # Smooth prediction
            recent_preds.append(pred)
            if len(recent_preds) > N:
                recent_preds.pop(0)
            smooth_pred = Counter(recent_preds).most_common(1)[0][0]
            activity_log.append(smooth_pred)

            # Draw landmarks and prediction text
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, POSE_CONNECTIONS)

            # Draw activity in red, large, bold
            cv2.putText(
                frame,
                f"Activity: {smooth_pred}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),  # Red color
                3,  # Thickness
                cv2.LINE_AA
            )
            stframe.image(frame, channels="BGR")

    cap.release()

    st.subheader("🧾 Activity Summary")
    summary = Counter(activity_log)
    total = sum(summary.values())
    for act, count in summary.items():
        st.write(f"**{act}**: {count} frames ({(count / total) * 100:.1f}%)")
    st.write("👉 The model uses pose data plus simple rules (leg angles & hand movement) to improve accuracy.")
else:
    st.info("Please upload a video file to start.")
