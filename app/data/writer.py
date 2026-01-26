import os
import numpy as np
from app.config import DATA_PATH

def save_sequence(action, sequence_data, modality='camera'):
    """Save a sequence of data for a specific action.
    
    Args:
        action (str): The label of the action.
        sequence_data (np.array): Timing series data (e.g., landmarks or sensor data).
        modality (str): 'camera' or 'glove'.
    """
    action_path = os.path.join(DATA_PATH, modality, action)
    os.makedirs(action_path, exist_ok=True)
    
    # Find the next sequence number (only looking at folders that are numbers)
    existing_sequences = [d for d in os.listdir(action_path) 
                          if os.path.isdir(os.path.join(action_path, d)) and d.isdigit()]
    sequence_num = 0
    if existing_sequences:
        sequence_num = max([int(s) for s in existing_sequences]) + 1
    
    sequence_dir = os.path.join(action_path, str(sequence_num))
    os.makedirs(sequence_dir, exist_ok=True)
    
    # Save each frame/timestep as a separate npy file (consistent with old notebook)
    # or save the whole sequence as one file? 
    # The requirement says "Append-only" and "Raw data is the source of truth".
    # Old notebook saved frames individually: action/sequence/frame.npy
    
    for frame_num, frame_data in enumerate(sequence_data):
        np.save(os.path.join(sequence_dir, f"{frame_num}.npy"), frame_data)
        
    print(f"Saved sequence {sequence_num} for action '{action}' in {modality}")
