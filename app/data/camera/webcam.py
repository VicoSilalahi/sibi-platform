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
        
        # Threading Queue
        self.frame_queue = queue.Queue(maxsize=1)
        
        # Latest data storage for UI access
        self.latest_raw_frame = None
        self.latest_processed_data = (None, None, None) # (image, landmarks, results)
        self.data_lock = threading.Lock()
        
        # ROI Tracking State (Thread-local effectively)
        self.last_roi = None

    def _capture_thread(self):
        """Thread for capturing frames from camera."""
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                break
                
            ret, frame = self.cap.read()
            if not ret:
                break
                
            with self.data_lock:
                self.latest_raw_frame = frame.copy()
            
            # Non-blocking put
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def _process_thread(self):
        """Thread for MediaPipe processing."""
        from app.config import MP_MODEL_COMPLEXITY, MP_STATIC_IMAGE_MODE
        mp_hands, _ = _get_mp()
        
        with mp_hands.Hands(
            static_image_mode=MP_STATIC_IMAGE_MODE,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            max_num_hands=2
        ) as hands:
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Pass self.last_roi for temporal continuity
                    try:
                         # mediapipe_detection now expects 'model' to be a Hands instance
                         image, results, new_roi = mediapipe_detection(frame, hands, self.last_roi)
                         self.last_roi = new_roi
                         
                         landmarks = extract_keypoints(results)
                         
                         with self.data_lock:
                             self.latest_processed_data = (image, landmarks, results)
                             
                    except Exception as inner_e:
                        import traceback
                        traceback.print_exc()
                        print(f"Frame Process Error: {inner_e}")
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Process Thread Error: {e}")

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
        if hasattr(self, 't_cap') and self.t_cap.is_alive(): self.t_cap.join()
        if hasattr(self, 't_proc') and self.t_proc.is_alive(): self.t_proc.join()
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def get_latest(self):
        """Get the latest raw frame and latest processed data without blocking."""
        with self.data_lock:
            return self.latest_raw_frame, self.latest_processed_data
