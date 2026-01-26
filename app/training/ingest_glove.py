import os
import csv
import argparse
import numpy as np
import shutil
from app.data.writer import save_sequence
from app.config import ACTIONS, DATA_PATH

def ingest_glove(input_dir='raw_glove'):
    """Ingest raw glove CSV logs into the landmark dataset."""
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found.")
        return

    # Process each action subdirectory
    for action in os.listdir(input_dir):
        if action not in ACTIONS:
            if action != "processed":
                print(f"Skipping unknown action: {action}")
            continue
            
        action_dir = os.path.join(input_dir, action)
        if not os.path.isdir(action_dir):
            continue
            
        print(f"Processing glove logs for action: '{action}'")
        processed_dir = os.path.join(action_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Get and sort CSV files numerically
        files = [f for f in os.listdir(action_dir) if f.endswith('.csv')]
        
        def get_file_num(filename):
            name = os.path.splitext(filename)[0]
            try:
                return int(name)
            except ValueError:
                return filename
                
        files.sort(key=get_file_num)

        for log_file in files:
            log_path = os.path.join(action_dir, log_file)
            sequence_data = []
            
            try:
                with open(log_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        sequence_data.append([float(val) for val in row])
                
                if len(sequence_data) > 0:
                    save_sequence(action, np.array(sequence_data), modality='glove')
                    # Move to processed folder
                    shutil.move(log_path, os.path.join(processed_dir, log_file))
                    print(f"Moved {log_file} to processed/")
                else:
                    print(f"Empty log file: {log_file}")
            except Exception as e:
                print(f"Error processing {log_file}: {e}")

        print(f"Finished processing glove logs for '{action}'")

def main():
    parser = argparse.ArgumentParser(description="Ingest Raw Glove CSVs into SIBI Dataset")
    parser.add_argument("--indir", type=str, default="raw_glove", help="Directory containing action subfolders with CSVs")
    
    args = parser.parse_args()
    ingest_glove(args.indir)

if __name__ == "__main__":
    main()
