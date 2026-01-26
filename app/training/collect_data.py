import cv2
import numpy as np
import time
import argparse
import os
from app.config import ACTIONS, SEQUENCE_LENGTH, GRACE_PERIOD_SECONDS
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
        # Grace period with countdown
        for i in range(GRACE_PERIOD_SECONDS, 0, -1):
            # We need to pull frames to show the countdown on top of live video
            # Using a simplified loop for the countdown
            start_wait = time.time()
            while time.time() - start_wait < 1:
                # Get one frame to display
                try:
                    image, landmarks, results = next(webcam.get_frames())
                    cv2.putText(image, f"GET READY: {i}", (200, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                    cv2.imshow("Data Collection", image)
                    cv2.waitKey(1)
                except StopIteration:
                    break
        
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

def collect_both_data(action, num_samples):
    """Synchronized collection of both camera and glove data."""
    webcam = WebcamHandler()
    reader = GloveReader()
    webcam.start()
    
    print(f"Collecting {num_samples} HYBRID samples (Camera + Glove) for action: '{action}'")
    
    for sample_num in range(num_samples):
        # Grace period with countdown
        for i in range(GRACE_PERIOD_SECONDS, 0, -1):
            start_wait = time.time()
            while time.time() - start_wait < 1:
                try:
                    image, landmarks, results = next(webcam.get_frames())
                    cv2.putText(image, f"GET READY: {i}", (200, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                    cv2.imshow("Hybrid Collection", image)
                    cv2.waitKey(1)
                except StopIteration:
                    break
        
        camera_sequence = []
        glove_sequence = []
        print(f"Recording Sample {sample_num}...")
        
        count = 0
        for image, landmarks, results in webcam.get_frames():
            # Synchronously read glove frame
            glove_frame = reader.read_frame()
            
            # Overlay info
            cv2.putText(image, f"ACTION: {action} | SAMPLE: {sample_num}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"HYBRID FRAME: {count}/{SEQUENCE_LENGTH}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            draw_styled_landmarks(image, results)
            cv2.imshow("Hybrid Collection", image)
            
            camera_sequence.append(landmarks)
            glove_sequence.append(glove_frame)
            count += 1
            
            if count >= SEQUENCE_LENGTH:
                break
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                webcam.stop()
                return

        # Save both
        save_sequence(action, camera_sequence, modality='camera')
        save_sequence(action, glove_sequence, modality='glove')
        print(f"Sample {sample_num} (Hybrid) saved. Break...")
        time.sleep(1)

    webcam.stop()

def main():
    parser = argparse.ArgumentParser(description="SIBI Data Collection Tool")
    parser.add_argument("--action", type=str, required=True, choices=ACTIONS,
                        help="Action to record")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to record")
    parser.add_argument("--modality", type=str, default="camera", choices=["camera", "glove", "both"],
                        help="Modality: camera, glove, or both")
    
    args = parser.parse_args()
    
    if args.modality == "camera":
        collect_camera_data(args.action, args.samples)
    elif args.modality == "glove":
        collect_glove_data(args.action, args.samples)
    elif args.modality == "both":
        collect_both_data(args.action, args.samples)

if __name__ == "__main__":
    main()
