import cv2
import os
from app.data.camera.extractor import mediapipe_detection, extract_keypoints, _get_mp

class VideoLoader:
    def __init__(self, video_dir):
        self.video_dir = video_dir

    def get_video_landmarks(self, video_path):
        """Extract landmarks from a single video file."""
        landmarks_sequence = []
        cap = cv2.VideoCapture(video_path)
        
        mp_holistic, _ = _get_mp()
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                _, results = mediapipe_detection(frame, holistic)
                landmarks = extract_keypoints(results)
                landmarks_sequence.append(landmarks)
                
        cap.release()
        return landmarks_sequence

    def process_directory(self, callback):
        """Process all videos in the directory and call callback with results."""
        files = [v for v in os.listdir(self.video_dir) if v.endswith(('.mp4', '.avi', '.mov'))]
        
        # Sort files numerically if they follow the pattern "number.ext"
        def get_file_num(filename):
            name = os.path.splitext(filename)[0]
            try:
                return int(name)
            except ValueError:
                return filename # Fallback to string for non-numeric names
                
        files.sort(key=get_file_num)

        for video_file in files:
            video_path = os.path.join(self.video_dir, video_file)
            # Assuming filename or parent dir determines action label
            # This depends on user's dataset structure
            # For now, just extract and return
            landmarks = self.get_video_landmarks(video_path)
            callback(video_file, landmarks)
