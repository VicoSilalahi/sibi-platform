import argparse
import cv2
import numpy as np
from app.data.camera.webcam import WebcamHandler
from app.inference.camera_infer import CameraInference
from app.inference.glove_infer import GloveInference
from app.data.glove.reader import GloveReader
from app.models.utils import load_model
from app.ui import render_ui

def run_camera_inference(args):
    """Run real-time inference using the webcam."""
    model = load_model("camera_gru")
    if model is None:
        print("Camera model not found. Please run 'python -m app.training.train_camera' first.")
        return

    infer = CameraInference(model)
    
    retry_count = 0
    max_retries = 5
    webcam = None
    
    while retry_count < max_retries:
        try:
            webcam = WebcamHandler()
            webcam.start()
            break
        except Exception as e:
            print(f"Error starting webcam: {e}. Retrying in 2s...")
            retry_count += 1
            time.sleep(2)
            
    if not webcam:
        print("Fatal: Could not start webcam after multiple retries.")
        return

    print("Initializing MediaPipe & AI Engine...")
    
    last_landmarks = None
    label, confidence = "No sign", 0.0
    ai_ready = False
    
    import time
    from app.config import UI_FPS
    
    while True:
        loop_start = time.time()
        
        # Get latest data WITHOUT blocking
        image, (proc_img, landmarks, results) = webcam.get_latest()
        
        if image is None:
            time.sleep(0.01)
            continue
            
        # UI Overlay Logic
        status_text = "AI Ready" if ai_ready else "Loading AI Engine..."
        
        # Only run inference if we have NEW landmarks
        if landmarks is not None and landmarks is not last_landmarks:
            if not ai_ready:
                print("✅ AI Engine Ready & Processing.")
                ai_ready = True
            
            label, confidence = infer.process_frame(landmarks)
            last_landmarks = landmarks
            
            # Brief visual feedback for active inference
            status_text = f"Analyzing: {label}" if label != "No sign" else "Monitoring..."
        
        if not args.headless:
            display_image = render_ui(image, label, confidence)
            
            # Optional: Add status bar at the bottom
            cv2.putText(display_image, status_text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("SIBI Platform - Camera", display_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Headless mode: just log if something interesting happens
            if label != "No sign" and label != "No sign": # Duplicate check just to be sure
                print(f"[HEADLESS] Detected: {label} ({confidence:.2f})")
            
            # Allow clean exit via Ctrl+C which is handled by KeyboardInterrupt usually, 
            # but here we are in a loop.
            time.sleep(0.01)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # UI Throttling: Ensure we don't exceed target UI_FPS
        elapsed = time.time() - loop_start
        wait_time = max(0, (1.0 / UI_FPS) - elapsed)
        if wait_time > 0:
            time.sleep(wait_time)
            
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
            label, confidence = infer.process_frame(sensor_data)
            
            if label != "No sign":
                print(f"Detected: {label} ({int(confidence*100)}%)")
    except KeyboardInterrupt:
        print("Stopped.")

def main():
    parser = argparse.ArgumentParser(description="SIBI Platform Dynamic Sign Language Recognition")
    parser.add_argument("--mode", type=str, default="camera", choices=["camera", "glove"],
                        help="Operation mode: camera or glove")
    
    parser.add_argument("--headless", action="store_true", help="Run without UI (for services)")
    args = parser.parse_args()

    if args.mode == "camera":
        run_camera_inference(args)
    elif args.mode == "glove":
        run_glove_inference()

if __name__ == "__main__":
    main()
