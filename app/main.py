import argparse
import cv2
import numpy as np
from app.data.camera.webcam import WebcamHandler
from app.inference.camera_infer import CameraInference
from app.inference.glove_infer import GloveInference
from app.data.glove.reader import GloveReader
from app.models.utils import load_model
from app.ui import render_ui

def run_camera_inference():
    """Run real-time inference using the webcam."""
    model = load_model("camera_gru")
    if model is None:
        print("Camera model not found. Please run 'python -m app.training.train_camera' first.")
        return

    infer = CameraInference(model)
    webcam = WebcamHandler()
    webcam.start()

    print("Starting camera inference. Press 'q' to quit.")
    
    for image, landmarks, results in webcam.get_frames():
        label, confidence, is_active, energy = infer.process_frame(landmarks)
        
        # UI Overlay
        image = render_ui(image, label, confidence, is_active, energy)
        
        cv2.namedWindow("SIBI Platform - Camera", cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("SIBI Platform - Camera", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.stop()

def run_glove_inference():
    """Run real-time inference using the glove sensors."""
    model = load_model("glove_gru")
    if model is None:
        print("Glove model not found. Please run 'python -m app.training.train_glove' first.")
        return

    infer = GloveInference(model)
    reader = GloveReader()
    
    print("Starting glove inference. Press Ctrl+C to quit.")
    try:
        while True:
            sensor_data = reader.read_frame()
            label, confidence, is_active, energy = infer.process_frame(sensor_data)
            
            status = "[REC]" if is_active else "[IDLE]"
            if label != "No sign":
                print(f"{status} (Energy: {energy:.4f}) Detected: {label} ({int(confidence*100)}%)")
            else:
                print(f"{status} Energy: {energy:.4f}                       ", end="\r")
            
    except KeyboardInterrupt:
        print("\nStopped.")

def main():
    parser = argparse.ArgumentParser(description="SIBI Platform Dynamic Sign Language Recognition")
    parser.add_argument("--mode", type=str, default="camera", choices=["camera", "glove"],
                        help="Operation mode: camera or glove")
    
    args = parser.parse_args()

    if args.mode == "camera":
        run_camera_inference()
    elif args.mode == "glove":
        run_glove_inference()

if __name__ == "__main__":
    main()
