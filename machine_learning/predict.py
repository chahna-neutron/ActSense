from collections import deque
import joblib
import numpy as np
from machine_learning.feature_extraction import SEQUENCE_LENGTH, extract_sequence_features

MODEL_PATH = "machine_learning/models/activity_rf.pkl"


def format_landmarks(landmarks):
    """
    Accepts either:
      - Member 1's list of 33 dicts: [{"x": ..., "y": ..., "z": ..., "visibility": ...}, ...]
      - A flat list/array of 132 floats
    Returns a 1D numpy float32 array of length 132.
    """
    if len(landmarks) == 33 and isinstance(landmarks[0], dict):
        return np.array(
            [[lm["x"], lm["y"], lm["z"], lm.get("visibility", 0.0)] for lm in landmarks],
            dtype=np.float32
        ).flatten()
    return np.asarray(landmarks, dtype=np.float32).flatten()


class ActivityPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

    def reset(self):
        self.frame_buffer.clear()

    def predict_frame(self, landmarks):
        """
        Accepts 1 frame of landmarks (either 33 dicts from PoseDetector or 132 floats).
        Returns (activity_name: str, confidence: float).
        """
        flat_132 = format_landmarks(landmarks)
        self.frame_buffer.append(flat_132)

        if len(self.frame_buffer) < SEQUENCE_LENGTH:
            return "Collecting frames...", 0.0

        features = extract_sequence_features(list(self.frame_buffer)).reshape(1, -1)
        probs = self.model.predict_proba(features)[0]
        best_idx = int(np.argmax(probs))

        activity = str(self.model.classes_[best_idx])
        confidence = round(float(probs[best_idx]), 2)
        return activity, confidence


class MultiPersonActivityPredictor:
    """Maintains a separate 30-frame buffer per tracked person_id from Member 1's PoseDetector."""
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.predictors = {}

    def predict_people(self, people_data):
        """
        Takes detection_data["people"] from PoseDetector.process_frame(frame).
        Returns a dict: {person_id: (activity, confidence)}
        """
        active_ids = {p["person_id"] for p in people_data}
        for pid in list(self.predictors.keys()):
            if pid not in active_ids:
                del self.predictors[pid]

        results = {}
        for person in people_data:
            pid = person["person_id"]
            if pid not in self.predictors:
                self.predictors[pid] = ActivityPredictor(self.model_path)
            results[pid] = self.predictors[pid].predict_frame(person["landmarks"])
        return results