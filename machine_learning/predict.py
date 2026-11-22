from collections import deque
import joblib
import numpy as np
from machine_learning.feature_extraction import SEQUENCE_LENGTH, extract_sequence_features

MODEL_PATH = "machine_learning/models/activity_rf.pkl"


class ActivityPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

    def reset(self):
        self.frame_buffer.clear()

    def predict_frame(self, landmarks_132):
        """
        Accepts 1 frame of 33 MediaPipe landmarks flattened to 132 values [x, y, z, visibility, ...].
        Maintains a sliding window of the last 30 frames and returns (activity, confidence).
        """
        self.frame_buffer.append(np.asarray(landmarks_132, dtype=np.float32).flatten())

        if len(self.frame_buffer) < SEQUENCE_LENGTH:
            return "Collecting frames...", 0.0

        features = extract_sequence_features(list(self.frame_buffer)).reshape(1, -1)
        probs = self.model.predict_proba(features)[0]
        best_idx = int(np.argmax(probs))

        activity = str(self.model.classes_[best_idx])
        confidence = round(float(probs[best_idx]), 2)
        return activity, confidence


# Quick verification test
if __name__ == "__main__":
    predictor = ActivityPredictor()
    for _ in range(SEQUENCE_LENGTH):
        activity, confidence = predictor.predict_frame(np.zeros(132))
    print(f"Integration test passed -> Activity: {activity}, Confidence: {confidence}")