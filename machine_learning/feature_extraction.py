import numpy as np

SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 33
VALUES_PER_LANDMARK = 4
FRAME_FEATURE_SIZE = NUM_LANDMARKS * VALUES_PER_LANDMARK  # 132

def extract_sequence_features(sequence_frames):
    """
    Converts 30 consecutive frames of 33 MediaPipe landmarks into a single 1D feature vector.

    Args:
        sequence_frames: array-like of shape (30, 132) or flat array of length 3960.
                         Each frame has 33 landmarks * [x, y, z, visibility].

    Returns:
        1D numpy array of extracted posture and motion features.
    """
    seq = np.asarray(sequence_frames, dtype=np.float32).reshape(
        SEQUENCE_LENGTH, NUM_LANDMARKS, VALUES_PER_LANDMARK
    )

    coords = seq[:, :, :3]              # (30, 33, 3) -> x, y, z
    visibility = seq[:, :, 3].mean(axis=0)  # (33,) -> average landmark confidence

    # 1. Center coordinates around mid-hip (landmarks 23 & 24) so distance/position in frame doesn't matter
    hip_center = (coords[:, 23:24, :] + coords[:, 24:25, :]) / 2.0
    norm_coords = coords - hip_center

    # 2. Posture features: average normalized body shape across the 30 frames (99 features)
    mean_pose = norm_coords.mean(axis=0).flatten()

    # 3. Movement speed features: frame-to-frame joint velocity (99 + 99 = 198 features)
    frame_diffs = np.diff(norm_coords, axis=0)  # (29, 33, 3)
    mean_speed = np.abs(frame_diffs).mean(axis=0).flatten()
    max_speed = np.abs(frame_diffs).max(axis=0).flatten()

    # 4. Vertical motion features: raw hip Y-axis displacement & peak speed for Jumping (2 features)
    raw_hip_y = hip_center[:, 0, 1]
    vertical_motion = np.array([
        raw_hip_y.max() - raw_hip_y.min(),
        np.abs(np.diff(raw_hip_y)).max()
    ], dtype=np.float32)

    return np.concatenate([mean_pose, mean_speed, max_speed, vertical_motion, visibility])