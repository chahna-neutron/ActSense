import os
import math
import time
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class PoseDetector:
    def __init__(
        self,
        num_poses=5,
        enable_hands=True,
        num_hands=10,
        smoothing_alpha=0.7,
        max_missing_frames=20
    ):
        # Prefer full model for higher accuracy; fall back to lite if not downloaded
        full_model = os.path.join(PROJECT_ROOT, "models", "pose_landmarker_full.task")
        lite_model = os.path.join(PROJECT_ROOT, "models", "pose_landmarker_lite.task")
        pose_model = full_model if os.path.exists(full_model) else lite_model

        if not os.path.exists(pose_model):
            raise FileNotFoundError(
                f"Pose model not found in {os.path.join(PROJECT_ROOT, 'models')}"
            )

        print(f"Loaded pose model: {os.path.basename(pose_model)}")

        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=0.65,
            min_pose_presence_confidence=0.65,
            min_tracking_confidence=0.65
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        # Optional hand detection (ON by default for live demo; can disable during dataset capture)
        self.enable_hands = enable_hands
        self.hand_detector = None
        if enable_hands:
            hand_model = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
            if not os.path.exists(hand_model):
                raise FileNotFoundError(f"Hand model not found: {hand_model}")

            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6
            )
            self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        self.pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
        ]
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
            (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        # Persistent tracker state: {pid: {"center": (cx, cy), "missing": int, "landmarks": list}}
        self.tracked_people = {}
        self.next_person_id = 1
        self.smoothing_alpha = smoothing_alpha
        self.max_missing_frames = max_missing_frames

        self.last_timestamp_ms = 0
        self.prev_frame_time = time.monotonic()

    @staticmethod
    def _get_torso_center(pose):
        cx = (pose[11].x + pose[12].x + pose[23].x + pose[24].x) / 4.0
        cy = (pose[11].y + pose[12].y + pose[23].y + pose[24].y) / 4.0
        return cx, cy

    def _assign_persistent_ids(self, detected_poses):
        """Matches detected people to closest previous torso centres and holds IDs across brief flickers."""
        if not detected_poses:
            for pid in list(self.tracked_people.keys()):
                self.tracked_people[pid]["missing"] += 1
                if self.tracked_people[pid]["missing"] > self.max_missing_frames:
                    del self.tracked_people[pid]
            return []

        centers = [self._get_torso_center(pose) for pose in detected_poses]

        pairs = []
        for pid, state in self.tracked_people.items():
            px, py = state["center"]
            for idx, (cx, cy) in enumerate(centers):
                dist = math.hypot(cx - px, cy - py)
                if dist < 0.25:
                    pairs.append((dist, pid, idx))

        pairs.sort(key=lambda item: item[0])

        matched_pids = set()
        matched_indices = {}

        for _, pid, idx in pairs:
            if pid not in matched_pids and idx not in matched_indices:
                matched_pids.add(pid)
                matched_indices[idx] = pid

        for pid in list(self.tracked_people.keys()):
            if pid not in matched_pids:
                self.tracked_people[pid]["missing"] += 1
                if self.tracked_people[pid]["missing"] > self.max_missing_frames:
                    del self.tracked_people[pid]

        assigned = []
        for idx, pose in enumerate(detected_poses):
            if idx in matched_indices:
                pid = matched_indices[idx]
            else:
                pid = self.next_person_id
                self.next_person_id += 1
                self.tracked_people[pid] = {
                    "center": centers[idx],
                    "missing": 0,
                    "landmarks": None
                }

            self.tracked_people[pid]["center"] = centers[idx]
            self.tracked_people[pid]["missing"] = 0
            assigned.append((pid, pose))

        return assigned

    def _smooth_landmarks(self, pid, raw_landmarks):
        """Applies light exponential smoothing to reduce micro-jitter without lagging fast actions."""
        prev = self.tracked_people[pid].get("landmarks")
        if prev is None:
            smoothed = raw_landmarks
        else:
            a = self.smoothing_alpha
            smoothed = []
            for cur, old in zip(raw_landmarks, prev):
                smoothed.append({
                    "x": round(a * cur["x"] + (1 - a) * old["x"], 4),
                    "y": round(a * cur["y"] + (1 - a) * old["y"], 4),
                    "z": round(a * cur["z"] + (1 - a) * old["z"], 4),
                    "visibility": cur["visibility"]
                })

        self.tracked_people[pid]["landmarks"] = smoothed
        return smoothed

    @staticmethod
    def _joint_visible(landmarks, idx, threshold=0.5):
        p = landmarks[idx]
        return p["visibility"] >= threshold and 0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0

    def _is_full_body_visible(self, landmarks, threshold=0.5):
        """Checks for torso plus at least one visible leg (knee + ankle) so side-profile motion works."""
        shoulder_ok = self._joint_visible(landmarks, 11, threshold) or self._joint_visible(landmarks, 12, threshold)
        hip_ok = self._joint_visible(landmarks, 23, threshold) or self._joint_visible(landmarks, 24, threshold)
        left_leg_ok = self._joint_visible(landmarks, 25, threshold) and self._joint_visible(landmarks, 27, threshold)
        right_leg_ok = self._joint_visible(landmarks, 26, threshold) and self._joint_visible(landmarks, 28, threshold)

        return shoulder_ok and hip_ok and (left_leg_ok or right_leg_ok)

    def _draw_clean_skeleton(self, frame, landmarks, vis_threshold=0.5):
        """Draws only confident, in-frame joints and connections."""
        h, w, _ = frame.shape
        pixels = []

        for p in landmarks:
            px, py = int(p["x"] * w), int(p["y"] * h)
            visible = p["visibility"] >= vis_threshold and 0 <= px < w and 0 <= py < h
            pixels.append((px, py, visible))

        for start_idx, end_idx in self.pose_connections:
            x1, y1, v1 = pixels[start_idx]
            x2, y2, v2 = pixels[end_idx]
            if v1 and v2:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

        for px, py, visible in pixels:
            if visible:
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1, cv2.LINE_AA)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = time.monotonic_ns() // 1_000_000
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        pose_results = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
        detected_poses = pose_results.pose_landmarks if pose_results.pose_landmarks else []
        tracked_poses = self._assign_persistent_ids(detected_poses)

        people_data = []
        h, w, _ = frame.shape

        for person_id, raw_pose in tracked_poses:
            raw_landmarks = [
                {
                    "x": round(point.x, 4),
                    "y": round(point.y, 4),
                    "z": round(point.z, 4),
                    "visibility": round(getattr(point, "visibility", 0.0), 4)
                }
                for point in raw_pose
            ]

            landmarks = self._smooth_landmarks(person_id, raw_landmarks)
            full_body = self._is_full_body_visible(landmarks)

            self._draw_clean_skeleton(frame, landmarks)

            people_data.append({
                "person_id": person_id,
                "full_body_visible": full_body,
                "landmarks": landmarks
            })

            nose = landmarks[0]
            label = f"Person {person_id}" if full_body else f"Person {person_id} (Partial)"
            color = (0, 255, 0) if full_body else (0, 165, 255)
            label_x = max(10, min(w - 180, int(nose["x"] * w)))
            label_y = max(25, int(nose["y"] * h) - 20)

            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        hands_data = []
        if self.enable_hands and self.hand_detector:
            hand_results = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
            if hand_results.hand_landmarks:
                for hand_id, hand_landmarks in enumerate(hand_results.hand_landmarks):
                    hand_points = [
                        {"x": round(p.x, 4), "y": round(p.y, 4), "z": round(p.z, 4)}
                        for p in hand_landmarks
                    ]
                    pixels = [(int(p["x"] * w), int(p["y"] * h)) for p in hand_points]
                    for s, e in self.hand_connections:
                        cv2.line(frame, pixels[s], pixels[e], (255, 255, 0), 2, cv2.LINE_AA)
                    for px, py in pixels:
                        cv2.circle(frame, (px, py), 3, (0, 200, 255), -1, cv2.LINE_AA)

                    handedness = "Unknown"
                    if hand_results.handedness and hand_id < len(hand_results.handedness) and hand_results.handedness[hand_id]:
                        handedness = hand_results.handedness[hand_id][0].category_name

                    hands_data.append({
                        "hand_id": hand_id + 1,
                        "handedness": handedness,
                        "landmarks": hand_points
                    })

        now = time.monotonic()
        fps = 1.0 / max(now - self.prev_frame_time, 1e-6)
        self.prev_frame_time = now

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"People: {len(people_data)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        if self.enable_hands:
            cv2.putText(frame, f"Hands: {len(hands_data)}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

        return frame, {
            "timestamp_ms": timestamp_ms,
            "people": people_data,
            "hands": hands_data
        }

    def close(self):
        self.pose_detector.close()
        if self.hand_detector:
            self.hand_detector.close()