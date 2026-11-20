import cv2
from computer_vision.pose_detection import PoseDetector
from machine_learning.predict import MultiPersonActivityPredictor


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
        activity_tracker = MultiPersonActivityPredictor()

        print("ActSense AI camera started! Press 'Q' in the video window to exit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error reading camera frame.")
                break

            # Mirror frame for natural webcam view
            frame = cv2.flip(frame, 1)

            annotated_frame, detection_data = detector.process_frame(frame)
            predictions = activity_tracker.predict_people(detection_data["people"])

            # Display detected activity and confidence on screen
            y_offset = 145
            for pid, (activity, confidence) in predictions.items():
                text = f"Person {pid}: {activity} ({confidence * 100:.0f}%)"
                cv2.putText(
                    annotated_frame,
                    text,
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 100, 255),
                    2,
                    cv2.LINE_AA
                )
                y_offset += 35

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

            cv2.imshow("ActSense AI - Live Pose & Activity Tracking", annotated_frame)

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