import numpy as np
from app.config import ACTIONS, CONFIDENCE_THRESHOLD, TEMPORAL_STABILITY_FRAMES, SEQUENCE_LENGTH, INFERENCE_STRIDE
from app.inference.gating import is_glove_active
from app.voice import bicara_piper

class GloveInference:
    def __init__(self, model):
        self.model = model
        self.sequence = []
        self.predictions = []
        self.current_label = "No sign"
        self.current_confidence = 0.0
        self.frame_count = 0
        self.action_label_now = None

    def process_frame(self, sensor_data):
        """Process a single frame of sensor data and return the detected sign.
        
        Args:
            sensor_data (np.array): Shape (22,)
        Returns:
            tuple: (label, confidence, is_active, energy)
        """
        self.sequence.append(sensor_data)
        self.sequence = self.sequence[-SEQUENCE_LENGTH:] # Keep last frames
        self.frame_count += 1

        is_active = False
        energy = 0.0
        if len(self.sequence) == SEQUENCE_LENGTH:
            is_active, energy = is_glove_active(np.array(self.sequence))
            
            if (self.frame_count % INFERENCE_STRIDE == 0):
                # Check gating
                if not is_active:
                    self.current_label = "No sign"
                    self.current_confidence = 0.0
                else:
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
                                action_label = ACTIONS[action_idx]
                                if action_label != self.action_label_now:
                                    bicara_piper(ACTIONS[action_idx])
                                    self.action_label_now = action_label
                                    self.current_label = ACTIONS[action_idx]
                                    self.current_confidence = confidence
                            else:
                                self.current_label = "No sign"
                                self.current_confidence = 0.0
                                self.action_label_now = None

        return self.current_label, self.current_confidence, is_active, energy
