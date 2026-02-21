import cv2
import os
from app.data.camera.extractor import mediapipe_detection, extract_keypoints, _get_mp, draw_styled_landmarks
from app.config import ACTIVATE_DRAWING_POINT

class VideoLoader:
    def __init__(self, video_dir):
        self.video_dir = video_dir

    def get_video_landmarks(self, video_path):
        """Extract landmarks from a single video file."""
        landmarks_sequence = []
        cap = cv2.VideoCapture(video_path)
        
        from app.config import MP_MODEL_COMPLEXITY, MP_STATIC_IMAGE_MODE
        mp_holistic, mp_drawing = _get_mp()
        with mp_holistic.Holistic(
            static_image_mode=MP_STATIC_IMAGE_MODE,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.5
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                image, results = mediapipe_detection(frame, holistic)
                if ACTIVATE_DRAWING_POINT:
                    draw_styled_landmarks(image, results)
                landmarks = extract_keypoints(results)
                landmarks_sequence.append(landmarks)
                
        cap.release()
        return landmarks_sequence

    def process_directory(self, callback):
        """Process all videos in the directory and call callback with results."""
        from app.config import INGESTED_SUFFIX
        raw_files = [v for v in os.listdir(self.video_dir) if v.endswith(('.mp4', '.avi', '.mov'))]
        
        # Skip files that were already ingested
        files = [f for f in raw_files if INGESTED_SUFFIX not in f]
        
        # Robust sorting: 0.avi < 1.avi < 0_timestamp.avi
        def get_sort_key(filename):
            name = os.path.splitext(filename)[0]
            if "_" in name:
                # Handle timestamped collisions: [original_num, timestamp]
                parts = name.split('_')
                try:
                    return [int(p) for p in parts]
                except ValueError:
                    return [float('inf'), name]
            try:
                # Handle pure numbers: [num, 0]
                return [int(name), 0]
            except ValueError:
                # Fallback for anything else
                return [float('inf'), name]
                
        files.sort(key=get_sort_key)

        for video_file in files:
            video_path = os.path.join(self.video_dir, video_file)
            # Assuming filename or parent dir determines action label
            # This depends on user's dataset structure
            # For now, just extract and return
            landmarks = self.get_video_landmarks(video_path)
            callback(video_file, landmarks)
