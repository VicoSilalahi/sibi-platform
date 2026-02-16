import cv2
import os
import time
import argparse
from app.config import ACTIONS, SEQUENCE_LENGTH, GRACE_PERIOD_SECONDS

def record_raw_videos(action, num_samples, output_dir='raw_videos'):
    """Record raw videos for a specific action."""
    action_dir = os.path.join(output_dir, action)
    os.makedirs(action_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    # Set to a common resolution (e.g., 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    
    # Define codec and VideoWriter settings
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    print(f"Recording {num_samples} raw videos for action: '{action}'")
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    import socket
    hostname = socket.gethostname().replace(" ", "_")
    
    for sample_num in range(num_samples):
        # Generate unique filename: hostname_timestamp.avi
        timestamp = int(time.time())
        video_name = f"{hostname}_{timestamp}.avi"
        video_path = os.path.join(action_dir, video_name)
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        # Grace period with countdown
        for i in range(GRACE_PERIOD_SECONDS, 0, -1):
            start_wait = time.time()
            while time.time() - start_wait < 1:
                ret, frame = cap.read()
                if not ret: break
                display_frame = frame.copy()
                cv2.putText(display_frame, f"GET READY: {i}", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                cv2.imshow("Record Raw Videos", display_frame)
                cv2.waitKey(1)

        print(f"Recording Video {sample_num + 1} of {num_samples}...")
        
        frame_count = 0
        while frame_count < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Show live feedback with frame count
            display_frame = frame.copy()
            cv2.putText(display_frame, f"RECORDING: {action} | {frame_count}/{SEQUENCE_LENGTH}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Record Raw Videos", display_frame)
            
            out.write(frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        out.release()
        print(f"Saved {video_path}")
        time.sleep(1) # Small break between samples
        
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Record Raw Videos for SIBI Dataset")
    parser.add_argument("--action", type=str, required=True, choices=ACTIONS, help="Action to record")
    parser.add_argument("--samples", type=int, default=1, help="Number of videos to record")
    parser.add_argument("--outdir", type=str, default="raw_videos", help="Directory to save videos")
    
    args = parser.parse_args()
    record_raw_videos(args.action, args.samples, args.outdir)

if __name__ == "__main__":
    main()
