import numpy as np
from app.config import MOTION_ENERGY_THRESHOLD, IDLE_SENSOR_VARIANCE

def is_camera_active(window):
    """Check if there is sufficient motion in the camera landmarks.
    
    Args:
        window (np.array): Shape (SEQUENCE_LENGTH, 75, 3)
    Returns:
        tuple: (bool, float) - (is_active, energy)
    """
    if len(window) < 2:
        return False, 0.0
        
    # Calculate variance of landmarks across time
    # Focus on hand landmarks (index 33 to 75) as pose often has jitter
    hand_landmarks = window[:, 33:, :]
    variance = np.var(hand_landmarks, axis=0)
    avg_variance = np.mean(variance)
    
    return avg_variance > MOTION_ENERGY_THRESHOLD, avg_variance

def is_glove_active(window):
    """Check if there is sufficient variance in the glove sensors.
    
    Args:
        window (np.array): Shape (SEQUENCE_LENGTH, 22)
    Returns:
        tuple: (bool, float) - (is_active, energy)
    """
    if len(window) < 2:
        return False, 0.0
        
    variance = np.var(window, axis=0)
    avg_variance = np.mean(variance)
    
    return avg_variance > IDLE_SENSOR_VARIANCE, avg_variance
