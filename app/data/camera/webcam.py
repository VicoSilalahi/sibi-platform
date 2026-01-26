import cv2
import threading
import queue
from app.data.camera.extractor import mediapipe_detection, extract_keypoints, _get_mp

class WebcamHandler:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.results_queue = queue.Queue(maxsize=1)

    def _capture_thread(self):
        """Thread for capturing frames from camera."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def _process_thread(self):
        """Thread for MediaPipe processing."""
        from app.config import MP_MODEL_COMPLEXITY, MP_STATIC_IMAGE_MODE
        mp_holistic, _ = _get_mp()
        with mp_holistic.Holistic(
            static_image_mode=MP_STATIC_IMAGE_MODE,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        ) as holistic:
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=1)
                    image, results = mediapipe_detection(frame, holistic)
                    landmarks = extract_keypoints(results)
                    if not self.results_queue.full():
                        self.results_queue.put((image, landmarks, results))
                except queue.Empty:
                    continue

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.running = True
        self.t_cap = threading.Thread(target=self._capture_thread, daemon=True)
        self.t_proc = threading.Thread(target=self._process_thread, daemon=True)
        self.t_cap.start()
        self.t_proc.start()

    def stop(self):
        self.running = False
        if self.t_cap.is_alive(): self.t_cap.join()
        if self.t_proc.is_alive(): self.t_proc.join()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def get_frames(self):
        """Generator that yields the latest available processed frame."""
        while self.running:
            try:
                # Get the latest result
                image, landmarks, results = self.results_queue.get(timeout=1)
                yield image, landmarks, results
            except queue.Empty:
                continue
