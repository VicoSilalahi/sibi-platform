import os
import time
import argparse
import csv
import numpy as np
from app.config import ACTIONS, SEQUENCE_LENGTH, GRACE_PERIOD_SECONDS
from app.data.glove.reader import GloveReader

def record_raw_glove(action, num_samples, output_dir='raw_glove'):
    """Record raw glove sensor data to CSV files."""
    action_dir = os.path.join(output_dir, action)
    os.makedirs(action_dir, exist_ok=True)
    
    reader = GloveReader()
    
    print(f"Recording {num_samples} raw glove samples for action: '{action}'")
    
    import socket
    hostname = socket.gethostname().replace(" ", "_")
    
    for _ in range(num_samples):
        # Generate unique filename: hostname_timestamp.csv
        timestamp = int(time.time())
        log_name = f"{hostname}_{timestamp}.csv"
        log_path = os.path.join(action_dir, log_name)
        
        # Grace period with countdown
        print(f"\nRecording Sample {log_num} in {GRACE_PERIOD_SECONDS} seconds...")
        for i in range(GRACE_PERIOD_SECONDS, 0, -1):
            print(f"{i}...")
            time.sleep(1)
            
        print(f"REC >>>")
        
        sequence_data = []
        for i in range(SEQUENCE_LENGTH):
            frame = reader.read_frame()
            sequence_data.append(frame)
            # Small delay to simulate 30Hz if reader doesn't block
            # In a real scenario, Serial.read() blocks until data arrives.
            time.sleep(0.033) 
            
        # Save to CSV
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(sequence_data)
            
        print(f"Saved {log_path}")
        time.sleep(1) # Small break between samples
        
    print("\nRecording complete.")

def main():
    parser = argparse.ArgumentParser(description="Record Raw Glove Data for SIBI Dataset")
    parser.add_argument("--action", type=str, required=True, choices=ACTIONS, help="Action to record")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to record")
    parser.add_argument("--outdir", type=str, default="raw_glove", help="Directory to save logs")
    
    args = parser.parse_args()
    record_raw_glove(args.action, args.samples, args.outdir)

if __name__ == "__main__":
    main()
