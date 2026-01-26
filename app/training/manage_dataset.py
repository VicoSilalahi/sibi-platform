import os
import json
import argparse
import numpy as np
from app.config import ACTIONS, ACTIONS_FILE, DATA_PATH

def get_sample_counts(modality='camera'):
    """Count original and augmented samples per action for a given modality."""
    counts = {}
    modality_path = os.path.join(DATA_PATH, modality)
    
    for action in ACTIONS:
        action_path = os.path.join(modality_path, action)
        if os.path.exists(action_path):
            # Count directories in the action folder
            all_samples = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
            original = [d for d in all_samples if not d.startswith('aug_')]
            augmented = [d for d in all_samples if d.startswith('aug_')]
            counts[action] = {
                'original': len(original),
                'total': len(all_samples)
            }
        else:
            counts[action] = {'original': 0, 'total': 0}
    return counts

def list_actions():
    """Display current actions and their sample counts."""
    print("\n--- SIBI Dataset Overview ---")
    print(f"{'Action':<15} | {'Camera (Orig/Total)':<20} | {'Glove (Orig/Total)':<20}")
    print("-" * 60)
    
    cam_counts = get_sample_counts('camera')
    glove_counts = get_sample_counts('glove')
    
    for action in ACTIONS:
        c = cam_counts.get(action, {'original': 0, 'total': 0})
        g = glove_counts.get(action, {'original': 0, 'total': 0})
        
        cam_str = f"{c['original']:>3} ({c['total']:>3})"
        glove_str = f"{g['original']:>3} ({g['total']:>3})"
        
        print(f"{action:<15} | {cam_str:<20} | {glove_str:<20}")
    print(f"\nTotal Actions: {len(ACTIONS)}")

def add_action(new_action):
    """Add a new action to the list."""
    if new_action in ACTIONS:
        print(f"Action '{new_action}' already exists.")
        return
    
    updated_actions = list(ACTIONS)
    updated_actions.append(new_action)
    
    with open(ACTIONS_FILE, 'w') as f:
        json.dump(updated_actions, f, indent=4)
    print(f"Action '{new_action}' added successfully.")

def remove_action(action_to_remove):
    """Remove an action from the list."""
    if action_to_remove not in ACTIONS:
        print(f"Action '{action_to_remove}' not found.")
        return
    
    updated_actions = [a for a in ACTIONS if a != action_to_remove]
    
    with open(ACTIONS_FILE, 'w') as f:
        json.dump(updated_actions, f, indent=4)
    print(f"Action '{action_to_remove}' removed from config. Note: Data on disk was NOT deleted.")

def main():
    parser = argparse.ArgumentParser(description="SIBI Dataset Management Tool")
    parser.add_argument("--list", action="store_true", help="List actions and sample counts")
    parser.add_argument("--add", type=str, help="Add a new action")
    parser.add_argument("--remove", type=str, help="Remove an action from configuration")
    
    args = parser.parse_args()
    
    if args.add:
        add_action(args.add)
    elif args.remove:
        remove_action(args.remove)
    
    # Always list if no specific modify action or if --list is set
    if args.list or (not args.add and not args.remove):
        list_actions()

if __name__ == "__main__":
    main()
