import os
import numpy as np
from app.config import ACTIONS, DATA_PATH, SEQUENCE_LENGTH

def load_data(modality='camera'):
    """Load sequences and labels from the data directory.
    
    Returns:
        tuple: (X, y) where X is (samples, sequence_length, features...) 
               and y is (samples, num_actions)
    """
    label_map = {label: num for num, label in enumerate(ACTIONS)}
    sequences, labels = [], []
    
    modality_path = os.path.join(DATA_PATH, modality)
    if not os.path.exists(modality_path):
        print(f"Modality path {modality_path} does not exist.")
        return np.array([]), np.array([])

    for action in ACTIONS:
        action_path = os.path.join(modality_path, action)
        if not os.path.exists(action_path):
            continue
            
        for sequence in os.listdir(action_path):
            window = []
            sequence_dir = os.path.join(action_path, sequence)
            
            # Load exactly SEQUENCE_LENGTH frames
            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(sequence_dir, f"{frame_num}.npy")
                if os.path.exists(frame_path):
                    res = np.load(frame_path)
                    window.append(res)
                else:
                    # Pad if sequence is shorter (though writer should ensure full length)
                    if len(window) > 0:
                        window.append(np.zeros_like(window[0]))
                    else:
                        # Should not happen with valid data
                        pass
            
            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])

    X = np.array(sequences)
    
    import tensorflow as tf
    y = tf.keras.utils.to_categorical(labels, num_classes=len(ACTIONS)).astype(int)
    
    return X, y
