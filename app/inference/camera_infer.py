import numpy as np
from app.config import ACTIONS, CONFIDENCE_THRESHOLD, TEMPORAL_STABILITY_FRAMES, SEQUENCE_LENGTH, INFERENCE_STRIDE
from app.inference.gating import is_camera_active
from app.inference.filters import EMAFilter

class CameraInference:
    def __init__(self, model):
        self.model = model
        self.sequence = []
        self.predictions = []
        self.current_label = "No sign"
        self.current_confidence = 0.0
        self.frame_count = 0
        self.filter = EMAFilter(alpha=0.6) # Smooth landmarks

    def process_frame(self, landmarks):
        """Process a single frame and return the detected sign.
        
        Args:
            landmarks (np.array): Shape (75, 3)
        Returns:
            tuple: (label, confidence)
        """
        # Apply smoothing filter
        landmarks = self.filter.apply(landmarks)
        
        self.sequence.append(landmarks)
        self.sequence = self.sequence[-SEQUENCE_LENGTH:] # Keep last frames
        self.frame_count += 1

        if len(self.sequence) == SEQUENCE_LENGTH and (self.frame_count % INFERENCE_STRIDE == 0):
            # Check gating (Motion Energy)
            if not is_camera_active(np.array(self.sequence)):
                # If we were previously detecting something, log that it stopped due to lack of movement
                if self.current_label != "No sign":
                    print("[Gating] No motion detected, clearing label.")
                self.current_label = "No sign"
                self.current_confidence = 0.0
                return self.current_label, self.current_confidence

            # Predict
            print(".", end="", flush=True)
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
