import numpy as np
from app.config import ACTIONS, CONFIDENCE_THRESHOLD, TEMPORAL_STABILITY_FRAMES, SEQUENCE_LENGTH, INFERENCE_STRIDE
from app.inference.gating import is_camera_active

class CameraInference:
    def __init__(self, model):
        self.model = model
        self.sequence = []
        self.predictions = []
        self.current_label = "No sign"
        self.current_confidence = 0.0
        self.frame_count = 0

    def process_frame(self, landmarks):
        """Process a single frame and return the detected sign.
        
        Args:
            landmarks (np.array): Shape (75, 3)
        Returns:
            tuple: (label, confidence)
        """
        self.sequence.append(landmarks)
        self.sequence = self.sequence[-SEQUENCE_LENGTH:] # Keep last frames
        self.frame_count += 1

        if len(self.sequence) == SEQUENCE_LENGTH and (self.frame_count % INFERENCE_STRIDE == 0):
            # Check gating
            if not is_camera_active(np.array(self.sequence)):
                self.current_label = "No sign"
                self.current_confidence = 0.0
                return self.current_label, self.current_confidence

            # Predict
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
            action_idx = np.argmax(res)
            confidence = res[action_idx]
            
            self.predictions.append(action_idx)
            self.predictions = self.predictions[-TEMPORAL_STABILITY_FRAMES:]

            # Temporal Stability Check
            if len(self.predictions) == TEMPORAL_STABILITY_FRAMES:
                if all(p == action_idx for p in self.predictions):
                    if confidence > CONFIDENCE_THRESHOLD:
                        self.current_label = ACTIONS[action_idx]
                        self.current_confidence = confidence
                    else:
                        self.current_label = "No sign"
                        self.current_confidence = 0.0

        return self.current_label, self.current_confidence
