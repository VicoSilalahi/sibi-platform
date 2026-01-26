import cv2
from app.data.camera.extractor import mediapipe_detection, extract_keypoints, draw_styled_landmarks, _get_mp

class WebcamHandler:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def stop(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def get_frames(self):
        """Generator that yields (image, landmarks) for each frame."""
        mp_holistic, _ = _get_mp()
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                image, results = mediapipe_detection(frame, holistic)
                landmarks = extract_keypoints(results)
                
                yield image, landmarks, results

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
