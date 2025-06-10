import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import tempfile
from collections import Counter

# --- Constants and Setup ---
# Load trained model and scaler
MODEL_PATH = "model/activity_model.pkl"
SCALER_PATH = "model/scaler.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# MediaPipe Pose Setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS # Connections to draw on the pose

# Smoothing window size (number of frames)
N_FRAMES_SMOOTHING = 5

st.title("📹 Human Activity Recognition (Pose-Based)")

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

# --- Helper Functions ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points (e.g., for joints).

    Args:
        a: Coordinates of the first point.
        b: Coordinates of the second point (vertex of the angle).
        c: Coordinates of the third point.

    Returns:
        The angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_pose_features(results):
    """Extracts pose features from MediaPipe landmarks.

    Args:
        results: MediaPipe Pose results object.

    Returns:
        A tuple containing:
            - pose_array: Numpy array of pose landmarks (x, y, z, visibility) for the model.
                         Returns None if no landmarks are detected.
            - hip_angle: Average angle of the hips (degrees).
            - knee_angle: Average angle of the knees (degrees).
            - left_hand: Coordinates [x, y] of the left wrist.
            - right_hand: Coordinates [x, y] of the right wrist.
    """
    if not results.pose_landmarks:
        # Return default values if no landmarks are detected.
        # This ensures that the unpacking in the main loop doesn't fail.
        return None, 0, 0, [0, 0], [0, 0]

    landmarks = results.pose_landmarks.landmark

    # Landmark extraction and creation of pose_row:
    # Extract x, y, z coordinates and visibility for each landmark.
    # These are flattened into a single list `pose_row`.
    pose_row = []
    for lm in landmarks:
        pose_row.extend([lm.x, lm.y, lm.z, lm.visibility])
    # Convert to a NumPy array and reshape. The model expects a specific input shape.
    # We use the first 132 values, corresponding to 33 landmarks * 4 values each.
    pose_array = np.array(pose_row[:132]).reshape(1, -1)

    # Purpose of get_coords:
    # Helper function to simplify fetching 2D (x, y) coordinates of a landmark by its index.
    # MediaPipe landmarks are indexed, e.g., mp_pose.PoseLandmark.LEFT_KNEE.value.
    def get_coords(idx):
        """Extracts x, y coordinates of a landmark by its MediaPipe index."""
        lm = landmarks[idx]
        return [lm.x, lm.y]

    # Angle calculation (knee, hip):
    # Calculate angles of left/right knee and left/right hip using the `calculate_angle` function.
    # These angles are used for rule-based adjustments to the activity prediction.
    # For example, a small knee angle might indicate "Sitting".
    # Handling of exceptions or missing landmarks:
    # A try-except block is used to catch potential errors if some landmarks are not visible
    # (e.g., a person is partially out of frame). If an error occurs, default angle values
    # (180 degrees, implying a straight posture) and hand coordinates ([0,0]) are returned.
    try:
        # Using MediaPipe PoseLandmark enum for clarity and robustness against index changes.
        left_knee_pt_a = get_coords(mp_pose.PoseLandmark.LEFT_HIP.value)
        left_knee_pt_b = get_coords(mp_pose.PoseLandmark.LEFT_KNEE.value)
        left_knee_pt_c = get_coords(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        left_knee = calculate_angle(left_knee_pt_a, left_knee_pt_b, left_knee_pt_c)

        right_knee_pt_a = get_coords(mp_pose.PoseLandmark.RIGHT_HIP.value)
        right_knee_pt_b = get_coords(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        right_knee_pt_c = get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        right_knee = calculate_angle(right_knee_pt_a, right_knee_pt_b, right_knee_pt_c)

        left_hip_pt_a = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        left_hip_pt_b = get_coords(mp_pose.PoseLandmark.LEFT_HIP.value)
        left_hip_pt_c = get_coords(mp_pose.PoseLandmark.LEFT_KNEE.value)
        left_hip = calculate_angle(left_hip_pt_a, left_hip_pt_b, left_hip_pt_c)

        right_hip_pt_a = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        right_hip_pt_b = get_coords(mp_pose.PoseLandmark.RIGHT_HIP.value)
        right_hip_pt_c = get_coords(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        right_hip = calculate_angle(right_hip_pt_a, right_hip_pt_b, right_hip_pt_c)

        hip_angle = (left_hip + right_hip) / 2
        knee_angle = (left_knee + right_knee) / 2

        # Extraction of hand coordinates:
        # Get x, y coordinates for left and right wrists. These are used to detect
        # hand movement, which can indicate activities like "Working".
        left_hand = get_coords(mp_pose.PoseLandmark.LEFT_WRIST.value)
        right_hand = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST.value)
    except Exception as e:
        # If any landmark is not visible or an error occurs during angle calculation,
        # default to standing posture angles and zero hand coordinates.
        # This prevents crashes if some body parts are occluded.
        # st.warning(f"Error in landmark calculation: {e}. Using default values.") # Optional: show warning in UI
        hip_angle, knee_angle = 180, 180  # Default to standing straight
        left_hand, right_hand = [0, 0], [0, 0] # Default to no hand coordinates

    return pose_array, hip_angle, knee_angle, left_hand, right_hand

# --- Main Video Processing Logic ---
if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) # Create a temporary file to store the uploaded video
    tfile.write(video_file.read())
    video_path = tfile.name

    # Initialization Section for Video Processing
    cap = cv2.VideoCapture(video_path) # OpenCV video capture object
    stframe = st.empty() # Streamlit container to display video frames
    activity_log = [] # List to store activity predictions for summary

    # Variables for prediction smoothing
    recent_preds = [] # Stores the N most recent predictions
    # N_FRAMES_SMOOTHING is defined above with other constants

    # Variables for hand movement detection (for "Working" activity)
    prev_left_hand = None # Stores previous frame's left hand coordinates
    prev_right_hand = None # Stores previous frame's right hand coordinates

    # Initialize MediaPipe Pose model
    with mp_pose.Pose(static_image_mode=False,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7) as pose:

        # Video Loop: Process each frame of the video
        while cap.isOpened():
            ret, frame = cap.read() # Read a frame from the video
            if not ret:
                # If no frame is returned (end of video), break the loop.
                break

            # Frame Reading and Pose Detection
            # Convert frame from BGR (OpenCV default) to RGB (MediaPipe default)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame with MediaPipe Pose to detect landmarks
            results = pose.process(image_rgb)

            # Feature Extraction: Call the helper function to get pose features
            features, hip_angle, knee_angle, left_hand, right_hand = extract_pose_features(results)

            current_pred = "No Pose" # Default prediction if no features

            if features is not None:
                # Model Prediction Section
                # Scale the extracted features using the loaded scaler
                features_scaled = scaler.transform(features)
                # Make a prediction using the loaded model
                model_prediction = model.predict(features_scaled)[0]
                current_pred = model_prediction # Use model's prediction initially

                # Rule-Based Activity Adjustment Section:
                # This section applies specific rules to refine the model's predictions
                # based on body angles or hand movements.

                # Sitting Rule: If knee angle is less than 100 degrees, classify as "Sitting".
                if knee_angle < 100:
                    current_pred = "Sitting"
                # Standing Rule: If knee angle is greater than 160 and hip angle greater than 150, classify as "Standing".
                elif knee_angle > 160 and hip_angle > 150:
                    current_pred = "Standing"

                # Working Rule: Detect repetitive hand movement.
                # This rule checks if hands have moved significantly from the previous frame,
                # and if the person is not sitting (knee_angle > 90 and hip_angle > 90).
                if prev_left_hand and prev_right_hand:
                    # Calculate Euclidean distance of hand movement
                    lh_move = np.linalg.norm(np.array(left_hand) - np.array(prev_left_hand))
                    rh_move = np.linalg.norm(np.array(right_hand) - np.array(prev_right_hand))
                    # If total hand movement exceeds a threshold and posture is not sitting, classify as "Working".
                    if (lh_move + rh_move) > 0.05 and knee_angle > 90 and hip_angle > 90:
                        current_pred = "Working"

                # Update previous hand positions for the next frame's movement calculation
                prev_left_hand = left_hand
                prev_right_hand = right_hand
            # else: features is None, current_pred remains "No Pose"

            # Prediction Smoothing Section:
            # Smooth out predictions over N_FRAMES_SMOOTHING frames to reduce flickering.
            recent_preds.append(current_pred)
            if len(recent_preds) > N_FRAMES_SMOOTHING:
                recent_preds.pop(0) # Remove the oldest prediction

            # Determine the most common prediction in the `recent_preds` list.
            smooth_pred = Counter(recent_preds).most_common(1)[0][0]
            activity_log.append(smooth_pred) # Log the smoothed prediction

            # Visualization Section:
            # Draw pose landmarks on the frame if detected.
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, POSE_CONNECTIONS)

            # Display the final smoothed activity prediction on the frame.
            cv2.putText(
                frame,
                f"Activity: {smooth_pred}",
                (30, 50), # Position of the text
                cv2.FONT_HERSHEY_SIMPLEX, # Font type
                1.5, # Font scale
                (0, 0, 255),  # Text color (Red in BGR)
                3,  # Text thickness
                cv2.LINE_AA # Line type
            )
            stframe.image(frame, channels="BGR") # Display the frame in Streamlit

    # Post-Loop Actions:
    cap.release() # Release the video capture object
    # No need to explicitly delete tfile here if using with statement for NamedTemporaryFile,
    # but since delete=False, manual cleanup might be needed depending on OS.
    # For this refactor, we assume original behavior is fine.

    # --- Activity Summary Section ---
    st.subheader("🧾 Activity Summary")
    # Count occurrences of each activity in the log.
    summary = Counter(activity_log)
    total_frames = sum(summary.values()) # Total number of processed frames

    # Display each activity and its percentage of occurrence.
    for act, count in summary.items():
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        st.write(f"**{act}**: {count} frames ({percentage:.1f}%)")

    st.write("👉 The model uses pose data plus simple rules (leg angles & hand movement) to improve accuracy.")
else:
    st.info("Please upload a video file to start.")

# General Code Hygiene:
# - Imports are already at the top.
# - Constants are grouped near the top.
# - Variable names seem descriptive.
# - Added blank lines for visual separation of logical blocks.
