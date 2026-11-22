import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd

ACTIVITIES = ["Standing", "Sitting", "Walking", "Running", "Jumping"]
SEQUENCE_LENGTH = 30    # 30 frames = ~1 second per clip
SAMPLES_PER_CLASS = 15  # 15 clips per activity = 75 total clips
DATA_PATH = "machine_learning/data/pose_sequences.csv"
TASK_MODEL_PATH = "machine_learning/models/pose_landmarker_lite.task"

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
]

os.makedirs("machine_learning/models", exist_ok=True)
if not os.path.exists(TASK_MODEL_PATH):
    print("Downloading MediaPipe Pose Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, TASK_MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.PoseLandmarker.create_from_options(options)

def extract_landmarks(result):
    if not result.pose_landmarks:
        return np.zeros(33 * 4, dtype=np.float32)
    landmarks = result.pose_landmarks[0]
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
        dtype=np.float32
    ).flatten()

def draw_skeleton(frame, result):
    if not result.pose_landmarks:
        return
    h, w, _ = frame.shape
    landmarks = result.pose_landmarks[0]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for idx in range(11, 33):
        cv2.circle(frame, points[idx], 4, (0, 255, 255), -1)
    for start_idx, end_idx in POSE_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

cap = cv2.VideoCapture(0)
all_rows = []
start_time_ms = int(time.time() * 1000)

print("=== ActSense AI Dataset Recorder ===")
print("Step back so your full body (shoulders to ankles) fits in the camera frame.")

for activity in ACTIVITIES:
    print(f"\nNext activity: {activity.upper()}")
    print("Click on the webcam window and press 's' when ready to record...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f"NEXT: {activity} | Press 's' to start", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("ActSense Data Collector", frame)
        if cv2.waitKey(10) & 0xFF == ord('s'):
            break

    for sample in range(SAMPLES_PER_CLASS):
        sequence = []
        for _ in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000) - start_time_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_skeleton(frame, result)
            sequence.append(extract_landmarks(result))

            cv2.putText(frame, f"REC {activity}: Clip {sample + 1}/{SAMPLES_PER_CLASS}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("ActSense Data Collector", frame)
            cv2.waitKey(1)

        if len(sequence) == SEQUENCE_LENGTH:
            row = list(np.array(sequence).flatten()) + [activity]
            all_rows.append(row)
        time.sleep(0.2)

cap.release()
landmarker.close()
cv2.destroyAllWindows()

os.makedirs("machine_learning/data", exist_ok=True)
df = pd.DataFrame(all_rows)
df.to_csv(DATA_PATH, index=False)
print(f"\nDone! Saved {len(df)} labeled sequences to {DATA_PATH}")