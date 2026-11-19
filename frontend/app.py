from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from machine_learning.predict import ActivityPredictor

st.set_page_config(page_title="ActSense AI", layout="wide")
st.title("ActSense AI - Human Activity Recognition")

ACTIVITIES = ["walking", "running", "sitting", "standing", "jumping"]

if "history" not in st.session_state:
    st.session_state.history = []
MODEL_PATH = "machine_learning/models/activity_rf.pkl"
POSE_TASK_PATH = "machine_learning/models/pose_landmark_lite.task"

if "predictor" not in st.session_state:
    st.session_state.predictor = ActivityPredictor(model_path=MODEL_PATH)


@st.cache_resource
def get_pose_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    task_file = ROOT_DIR / "machine_learning" / "models" / "pose_landmark_lite.task"
    model_bytes = task_file.read_bytes()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=VisionRunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_presence_confidence=0.3,
    )
    return PoseLandmarker.create_from_options(options)


def extract_landmarks_132(uploaded_file):
    if uploaded_file is None:
        return None

    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr_img is None:
        return None

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    landmarker = get_pose_landmarker()
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None

    landmarks = []
    for lm in result.pose_landmarks[0]:
        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(landmarks, dtype=np.float32)


def get_prediction(uploaded_file):
    landmarks_132 = extract_landmarks_132(uploaded_file)
    if landmarks_132 is None:
        return "No pose detected", 0.0

    activity, raw_confidence = st.session_state.predictor.predict_frame(landmarks_132)
    confidence_pct = round(float(raw_confidence) * 100, 1)
    return activity, confidence_pct


left, right = st.columns(2)

with left:
    st.subheader("Webcam")
    camera_img = st.camera_input("Camera")

with right:
    st.subheader("Prediction")
    if st.button("Predict Activity"):
        activity, confidence = get_prediction(camera_img)
        st.session_state.history.append(
            {"Activity": activity, "Confidence (%)": confidence}
        )

    if st.session_state.history:
        last = st.session_state.history[-1]
        st.metric("Activity", last["Activity"].upper())
        st.metric("Confidence", f"{last['Confidence (%)']}%")
        progress_val = max(0, min(100, int(last["Confidence (%)"])))
        st.progress(progress_val)

st.divider()

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Activity History")
        st.dataframe(df, use_container_width=True)
    with c2:
        st.subheader("Activity Graph")
        st.bar_chart(df["Activity"].value_counts())

    st.subheader("Confidence Over Time")
    st.line_chart(df["Confidence (%)"])