import numpy as np
from app.config import ACTIONS, CONFIDENCE_THRESHOLD, TEMPORAL_STABILITY_FRAMES, SEQUENCE_LENGTH, INFERENCE_STRIDE
from app.inference.gating import is_camera_active
from app.inference.filters import EMAFilter
import subprocess

from gtts import gTTS
import os

def bicara_piper(teks):
    # Ganti path model dengan file .onnx yang sudah kamu download
    model = "./piper/id_ID-news_tts-medium.onnx" 
    command = f'echo "{teks}" | ./piper/piper --model {model} --output_raw | aplay -r 22050 -f S16_LE -t raw'
    subprocess.run(command, shell=True)

class CameraInference:
    def __init__(self, model):
        self.model = model
        self.sequence = []
        self.predictions = []
        self.current_label = "No sign"
        self.current_confidence = 0.0
        self.frame_count = 0
        self.filter = EMAFilter(alpha=0.6) # Smooth landmarks
        self.start_time = None
        self.time_duration = 5.0
        self.action_label_now = None

    def process_frame(self, landmarks):
        """Process a single frame and return the detected sign.
        
        Args:
            landmarks (np.array): Shape (75, 3)
        Returns:
            tuple: (label, confidence, is_active, energy)
        """
        # Apply smoothing filter
        landmarks = self.filter.apply(landmarks)
        
        self.sequence.append(landmarks)
        self.sequence = self.sequence[-SEQUENCE_LENGTH:] # Keep last frames
        self.frame_count += 1

        is_active = False
        energy = 0.0
        if len(self.sequence) == SEQUENCE_LENGTH:
            is_active, energy = is_camera_active(np.array(self.sequence))
            
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
