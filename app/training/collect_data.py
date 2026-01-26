import cv2
import numpy as np
import time
import argparse
import os
from app.config import ACTIONS, SEQUENCE_LENGTH
from app.data.camera.webcam import WebcamHandler
from app.data.glove.reader import GloveReader
from app.data.writer import save_sequence
from app.data.camera.extractor import draw_styled_landmarks

def collect_camera_data(action, num_samples):
    webcam = WebcamHandler()
    webcam.start()
    
    print(f"Collecting {num_samples} samples for action: '{action}'")
    print("Prepare... starting in 3 seconds.")
    time.sleep(3)

    for sample_num in range(num_samples):
        sequence_data = []
        print(f"Recording Sample {sample_num}...")
        
        # Capture SEQUENCE_LENGTH frames
        count = 0
        for image, landmarks, results in webcam.get_frames():
            # Overlay info
            cv2.putText(image, f"ACTION: {action} | SAMPLE: {sample_num}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"FRAME: {count}/{SEQUENCE_LENGTH}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            draw_styled_landmarks(image, results)
            cv2.imshow("Data Collection", image)
            
            sequence_data.append(landmarks)
            count += 1
            
            if count >= SEQUENCE_LENGTH:
                break
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        save_sequence(action, sequence_data, modality='camera')
        print(f"Sample {sample_num} saved. Break...")
        time.sleep(1) # Small break between samples

    webcam.stop()

def collect_glove_data(action, num_samples):
    reader = GloveReader()
    print(f"Collecting {num_samples} samples for action: '{action}'")
    
    for sample_num in range(num_samples):
        print(f"Recording Sample {sample_num} in 2 seconds...")
        time.sleep(2)
        
        sequence_data = reader.get_sequence(SEQUENCE_LENGTH)
        save_sequence(action, sequence_data, modality='glove')
        print(f"Sample {sample_num} saved.")

def main():
    parser = argparse.ArgumentParser(description="SIBI Data Collection Tool")
    parser.add_argument("--action", type=str, required=True, choices=ACTIONS,
                        help="Action to record")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to record")
    parser.add_argument("--modality", type=str, default="camera", choices=["camera", "glove"],
                        help="Modality: camera or glove")
    
    args = parser.parse_args()
    
    if args.modality == "camera":
        collect_camera_data(args.action, args.samples)
    elif args.modality == "glove":
        collect_glove_data(args.action, args.samples)

if __name__ == "__main__":
    main()
