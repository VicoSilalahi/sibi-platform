import os
import numpy as np
import random
from app.config import ACTIONS, DATA_PATH, SEQUENCE_LENGTH

def add_jitter(sequence, amount=0.01):
    """Add Gaussian noise to landmarks."""
    noise = np.random.normal(0, amount, sequence.shape)
    return sequence + noise

def scale_sequence(sequence, factor_range=(0.9, 1.1)):
    """Randomly scale landmarks."""
    factor = random.uniform(*factor_range)
    return sequence * factor

def rotate_sequence(sequence, angle_range=(-5, 5)):
    """Randomly rotate landmarks around the center (rough approximation)."""
    angle = np.radians(random.uniform(*angle_range))
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    # Sequence is (30, 75, 3)
    return np.dot(sequence, rotation_matrix)

def augment_dataset(modality='camera', jitter=True, scale=True, rotate=True):
    modality_path = os.path.join(DATA_PATH, modality)
    if not os.path.exists(modality_path):
        print(f"Path {modality_path} not found.")
        return

    for action in ACTIONS:
        action_path = os.path.join(modality_path, action)
        if not os.path.exists(action_path):
            continue
            
        print(f"Augmenting action: {action}")
        sequences = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
        
        for seq_id in sequences:
            # Skip already augmented sequences
            if seq_id.startswith('aug_'):
                continue
                
            seq_dir = os.path.join(action_path, seq_id)
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(seq_dir, f"{frame_num}.npy")
                if os.path.exists(frame_path):
                    window.append(np.load(frame_path))
            
            if len(window) != SEQUENCE_LENGTH:
                continue
                
            window = np.array(window)
            
            # Create three augmentations per sequence
            for i in range(3):
                aug_window = window.copy()
                if jitter: aug_window = add_jitter(aug_window)
                if scale: aug_window = scale_sequence(aug_window)
                if rotate and modality == 'camera': aug_window = rotate_sequence(aug_window)
                
                # Save augmented sequence
                aug_id = f"aug_{seq_id}_{i}"
                aug_dir = os.path.join(action_path, aug_id)
                os.makedirs(aug_dir, exist_ok=True)
                for f_idx, f_data in enumerate(aug_window):
                    np.save(os.path.join(aug_dir, f"{f_idx}.npy"), f_data)

    print("Augmentation complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augment landmark dataset")
    parser.add_argument("--modality", type=str, default="camera", choices=["camera", "glove"])
    args = parser.parse_args()
    augment_dataset(modality=args.modality)
