import cv2
from computer_vision.pose_detection import PoseDetector


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access webcam. Check System Settings -> Privacy & Security -> Camera.")
        return

    detector = None

    try:
        detector = PoseDetector(
            num_poses=5,
            enable_hands=True,
            num_hands=10
        )

        print("ActSense AI camera started! Press 'Q' in the video window to exit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error reading camera frame.")
                break

            # Mirror frame for natural webcam view
            frame = cv2.flip(frame, 1)

            annotated_frame, detection_data = detector.process_frame(frame)

            # Bottom hint (keeps top-left clear for FPS, People, and Hands HUD)
            h = annotated_frame.shape[0]
            cv2.putText(
                annotated_frame,
                "Press Q to quit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            cv2.imshow("ActSense AI - Live Pose Tracking", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    finally:
        if detector is not None:
            detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Camera and detectors closed cleanly.")


if __name__ == "__main__":
    main()