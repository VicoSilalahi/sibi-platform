import time
import cv2
import numpy as np
import json
import argparse
from app.data.camera.webcam import WebcamHandler
from app.data.camera.extractor import mediapipe_detection, extract_keypoints, _get_mp
from app.inference.camera_infer import CameraInference
from app.models.utils import load_model
from app.config import ACTIONS, SEQUENCE_LENGTH, MP_MODEL_COMPLEXITY, MP_STATIC_IMAGE_MODE

class Profiler:
    def __init__(self):
        self.model = load_model("camera_gru")
        if self.model is None:
            raise FileNotFoundError("Camera model (camera_gru) not found.")

    def run_benchmark(self, num_frames=100):
        """Run performance benchmark (FPS/Latency)."""
        print(f"--- Profiling Inference ({num_frames} frames) ---")
        
        infer = CameraInference(self.model)
        webcam = WebcamHandler()
        webcam.start()

        latencies = {
            "ui_loop": [],
            "ai_update_interval": [], 
            "infer_logic": [],
        }

        last_landmarks = None
        last_ai_time = time.time()
        count = 0

        print("Collecting data...")
        try:
            # Warmup
            print("Warming up...")
            for _ in range(20):
                webcam.get_latest()
                time.sleep(0.05)

            while count < num_frames:
                loop_start = time.time()

                # Step 1: Get data
                image, (proc_img, landmarks, results) = webcam.get_latest()

                if image is None:
                    time.sleep(0.001)
                    continue

                latencies["ui_loop"].append((time.time() - loop_start) * 1000)

                # Step 2: Check if this is NEW AI data
                if landmarks is not None and landmarks is not last_landmarks:
                    now = time.time()
                    latencies["ai_update_interval"].append((now - last_ai_time) * 1000)
                    last_ai_time = now

                    infer_start = time.time()
                    label, confidence = infer.process_frame(landmarks)
                    latencies["infer_logic"].append((time.time() - infer_start) * 1000)

                    last_landmarks = landmarks
                    count += 1
                    if count % 20 == 0:
                        print(f"Captured {count}/{num_frames} fresh AI updates...")

                if image is not None:
                    cv2.imshow("Profiling (Benchmark)", image)
                    cv2.waitKey(1)

        finally:
            webcam.stop()
            cv2.destroyAllWindows()

        results = {
            "ui_latency_ms": np.mean(latencies['ui_loop']),
            "ui_fps": 1000/np.mean(latencies['ui_loop']),
            "ai_cycle_ms": np.mean(latencies['ai_update_interval']),
            "ai_fps": 1000/np.mean(latencies['ai_update_interval']),
            "infer_logic_ms": np.mean(latencies['infer_logic'])
        }

        print("\n--- Benchmark Results ---")
        print(f"UI Responsiveness:    {results['ui_latency_ms']:.2f}ms ({results['ui_fps']:.1f} FPS)")
        print(f"AI Cycle (Fresh MP):  {results['ai_cycle_ms']:.2f}ms ({results['ai_fps']:.1f} AI FPS)")
        print(f"GRU Inference Logic:  {results['infer_logic_ms']:.2f}ms")
        
        return results

    def run_batch_test(self):
        """Run interactive batch testing for accuracy verification."""
        print("--- SIBI Batch Test Mode ---")
        
        mp_holistic, _ = _get_mp()
        cap = cv2.VideoCapture(0)
        
        print("\nREADY: Press SPACE to record a sign.")
        
        with mp_holistic.Holistic(
            static_image_mode=MP_STATIC_IMAGE_MODE,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                display_frame = frame.copy()
                cv2.putText(display_frame, "READY - PRESS SPACE", (180, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Batch Test", display_frame)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # 1. RECORDING PHASE
                    frames = []
                    print("Recording...")
                    for i in range(SEQUENCE_LENGTH):
                        ret, frame = cap.read()
                        if not ret: break
                        frames.append(frame)
                        
                        prog_frame = frame.copy()
                        cv2.putText(prog_frame, f"REC: {len(frames)}/{SEQUENCE_LENGTH}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Batch Test", prog_frame)
                        cv2.waitKey(1)
                    
                    # 2. PROCESSING PHASE
                    print("Processing...")
                    start_process = time.time()
                    
                    landmarks_sequence = []
                    for f in frames:
                        _, results = mediapipe_detection(f, holistic)
                        landmarks = extract_keypoints(results)
                        landmarks_sequence.append(landmarks)
                    
                    # Inference part
                    if hasattr(self.model, 'predict'): # Keras
                        res = self.model.predict(np.expand_dims(landmarks_sequence, axis=0))[0]
                    else: # TFLite
                        res = self.model.predict(np.expand_dims(landmarks_sequence, axis=0))[0]
                            
                    action_idx = np.argmax(res)
                    confidence = res[action_idx]
                    
                    total_time = (time.time() - start_process) * 1000
                    
                    result_text = f"RESULT: {ACTIONS[action_idx]} ({int(confidence*100)}%)"
                    print(f"\n{result_text}")
                    print(f"Total Latency: {total_time:.1f}ms")
                    
                    # 3. DISPLAY RESULT
                    res_frame = frames[-1].copy()
                    cv2.rectangle(res_frame, (0,0), (640, 60), (0,0,0), -1)
                    cv2.putText(res_frame, result_text, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Batch Test", res_frame)
                    cv2.waitKey(3000)
                    print("\nREADY: Press SPACE to record again.")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark", "batch"], default="benchmark")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    
    profiler = Profiler()
    if args.mode == "benchmark":
        profiler.run_benchmark(args.frames)
    elif args.mode == "batch":
        profiler.run_batch_test()
