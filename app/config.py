import numpy as np
import os

# Suppress TensorFlow logs and oneDNN operations for faster/cleaner startup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('datasets')

# Dynamic Action Loading
import json
ACTIONS_FILE = 'actions.json'
if os.path.exists(ACTIONS_FILE):
    with open(ACTIONS_FILE, 'r') as f:
        ACTIONS = np.array(json.load(f))
else:
    ACTIONS = np.array(['yang', 'dan', 'dengan', 'ini', 'untuk'])

# Number of sequences per action (for training)
NO_SEQUENCES = 30

# Sequence length
SEQUENCE_LENGTH = 30

# Input Dimensions
CAMERA_LANDMARKS = 75  # 33 pose + 21 LH + 21 RH
CAMERA_INPUT_SHAPE = (SEQUENCE_LENGTH, CAMERA_LANDMARKS, 3)

GLOVE_SENSORS = 22  # (5 flex + 6 MPU) * 2
GLOVE_INPUT_SHAPE = (SEQUENCE_LENGTH, GLOVE_SENSORS)

# Inference Stability & Performance
INFERENCE_STRIDE = 3   # Run inference every N frames (1 = every frame, 5 = every 5 frames)
CONFIDENCE_THRESHOLD = 0.7
TEMPORAL_STABILITY_FRAMES = 5
MOTION_ENERGY_THRESHOLD = 0.01  # Minimum variance in landmarks to be considered active
IDLE_SENSOR_VARIANCE = 0.05    # Near zero variance for glove sensors

# Data Collection Settings
GRACE_PERIOD_SECONDS = 2  # Wait N seconds before each sample recording

# Path for saved models
MODEL_PATH = os.path.join('app', 'models', 'saved_models')
os.makedirs(MODEL_PATH, exist_ok=True)
