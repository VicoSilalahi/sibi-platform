import numpy as np
from app.config import ACTIONS, CONFIDENCE_THRESHOLD, TEMPORAL_STABILITY_FRAMES, SEQUENCE_LENGTH, INFERENCE_STRIDE
from app.inference.gating import is_camera_active
from app.inference.filters import EMAFilter

from gtts import gTTS
import os

def voice(text):
    print(f"Sedang memproses suara: '{text}'...")
    
    # 2. Proses Teks ke Suara Google (Bahasa Indonesia: 'id')
    tts = gTTS(text=text, lang='id')
    
    # 3. Simpan sementara sebagai mp3
    filename = "temp_voice.mp3"
    tts.save(filename)
    
    # 4. Putar menggunakan mpg123 (perintah sistem Arch)
    os.system(f"mpg123 -q {filename}")
    
    # Opsional: Hapus file setelah diputar
    os.remove(filename)

class CameraInference:
    def __init__(self, model):
        self.model = model
        self.sequence = []
        self.predictions = []
        self.current_label = "No sign"
        self.current_confidence = 0.0
        self.frame_count = 0
        self.filter = EMAFilter(alpha=0.6) # Smooth landmarks
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
                                    voice(ACTIONS[action_idx])
                                    self.action_label_now = action_label
                                    self.current_label = ACTIONS[action_idx]
                                    self.current_confidence = confidence
                            else:
                                self.current_label = "No sign"
                                self.current_confidence = 0.0
                                self.action_label_now = None

        return self.current_label, self.current_confidence, is_active, energy
