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

# Number of sequences per action (for training) without --samples flag this will be the default
NO_SEQUENCES = 30

# Sequence length (frames/data points)
SEQUENCE_LENGTH = 60

# Input Dimensions
CAMERA_LANDMARKS = 75  # MediaPipe Landmarks (75 points: 33 pose, 21 LH, 21 RH)
CAMERA_INPUT_SHAPE = (SEQUENCE_LENGTH, CAMERA_LANDMARKS, 3)

GLOVE_SENSORS = 22  # (5 flex + 6 MPU) * 2

# MediaPipe Optimization
# MediaPipe Optimization
MP_MODEL_COMPLEXITY = 0  # 0 = Lite (Fastest), 1 = Full, 2 = Heavy
MP_STATIC_IMAGE_MODE = False # False = treat as video stream
MP_INTERNAL_WIDTH = 160   # Downscale boost: Lower = Faster
MP_INTERNAL_HEIGHT = 120
MP_USE_CROP = True        # Enable ROI cropping
MP_CROP_SIZE = 400        
UI_FPS = 30               # Throttle UI to save CPU for AI
GLOVE_INPUT_SHAPE = (SEQUENCE_LENGTH, GLOVE_SENSORS)

# Inference Stability & Performance
INFERENCE_STRIDE = 1    # Run inference every fresh AI update (cheaper with TF-Lite)
INGESTED_SUFFIX = "_done" # Suffix to mark raw files as processed
CONFIDENCE_THRESHOLD = 0.7
TEMPORAL_STABILITY_FRAMES = 5
MOTION_ENERGY_THRESHOLD = 0.01  # Minimum variance in landmarks to be considered active
IDLE_SENSOR_VARIANCE = 0.05    # Near zero variance for glove sensors

# Data Collection Settings
GRACE_PERIOD_SECONDS = 3  # Wait N seconds before each sample recording

# Path for saved models
MODEL_PATH = os.path.join('app', 'models', 'saved_models')
os.makedirs(MODEL_PATH, exist_ok=True)
