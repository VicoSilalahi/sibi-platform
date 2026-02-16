import cv2
import os
import time
import argparse
import socket
import csv

# --- Configuration (Hardcoded for Independence) ---
# You can modify these to suit your needs
# https://pulaubahasa.wordpress.com/vocab-builders/pbwl/ at root
ACTIONS_CSV = 'actions.csv'
HARDCODED_ACTIONS = ['yang', 'dan', 'dengan', 'ini', 'untuk', 'dari', 'dalam', 'itu', 'tidak', 'pada', 'ada', 'akan', 'dapat', 'bagai', 'juga', 'adalah', 'ke', 'bisa', 'oleh', 'karena', 'telah', 'atau', 'lebih', 'saya', 'bagi', 'mereka', 'sudah', 'kita', 'cara', 'banyak', 'saat', 'anda', 'belum', 'harus', 'seperti', 'tetapi', 'kami', 'bahwa', 'kepada', 'para', 'dia', 'ia', 'jika', 'apa', 'semua', 'aku', 'sedang', 'tentang', 'kalau']

def get_actions():
    """Load actions from CSV or fallback to hardcoded list."""
    if os.path.exists(ACTIONS_CSV):
        try:
            with open(ACTIONS_CSV, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Assume actions are in the first column or a simple list
                actions = [row[0].strip() for row in reader if row and row[0].strip()]
                if actions:
                    return actions
        except Exception as e:
            print(f"Warning: Could not read {ACTIONS_CSV}: {e}")
    
    # If CSV doesn't exist or is empty, create it with hardcoded actions for convenience
    try:
        with open(ACTIONS_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for action in HARDCODED_ACTIONS:
                writer.writerow([action])
    except Exception as e:
        print(f"Warning: Could not create {ACTIONS_CSV}: {e}")
        
    return HARDCODED_ACTIONS

DEFAULT_SEQUENCE_LENGTH = 60
DEFAULT_GRACE_PERIOD = 3
DEFAULT_OUTPUT_DIR = 'raw_videos_test'

def record_raw_videos(action, num_samples, sequence_length, grace_period, output_dir):
    """Record raw videos for a specific action."""
    action_dir = os.path.join(output_dir, action)
    os.makedirs(action_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    # Set to a common resolution (e.g., 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Force 30 FPS
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Define codec and VideoWriter settings
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Get hostname for naming convention
    hostname = socket.gethostname().replace(" ", "_")
    
    print(f"\nTarget Action: '{action}'")
    print(f"Samples to record: {num_samples}")
    print(f"Sequence Length: {sequence_length} frames")
    print("-" * 30)
    
    for sample_num in range(num_samples):
        # Generate unique filename: hostname_timestamp.avi
        timestamp = int(time.time())
        video_name = f"{hostname}_{timestamp}.avi"
        video_path = os.path.join(action_dir, video_name)
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        # Grace period with countdown
        print(f"\nRecording Sample {sample_num + 1} of {num_samples} in {grace_period}s...")
        for i in range(grace_period, 0, -1):
            start_wait = time.time()
            while time.time() - start_wait < 1:
                ret, frame = cap.read()
                if not ret: break
                display_frame = frame.copy()
                cv2.putText(display_frame, f"GET READY: {i}", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                cv2.imshow("SIBI Recorder", display_frame)
                cv2.waitKey(1)

        print(f"REC >>> Recording {action}...")
        
        start_time = time.time()
        for i in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Show live feedback with frame count
            display_frame = frame.copy()
            cv2.putText(display_frame, f"RECORDING: {action} | {i+1}/{sequence_length}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("SIBI Recorder", display_frame)
            
            out.write(frame)
            
            # Maintain strictly 30fps
            elapsed = time.time() - start_time
            expected = (i + 1) / 30.0
            if expected > elapsed:
                time.sleep(expected - elapsed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nRecording cancelled by user.")
                out.release()
                cap.release()
                cv2.destroyAllWindows()
                return
        
        out.release()
        print(f"Saved: {video_path}")
        time.sleep(1) # Small break between samples
        
    print("\n" + "="*30)
    print("Recording Session Complete!")
    print("="*30)
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Standalone SIBI Video Recorder")
    parser.add_argument("--action", type=str, help="Action to record (overrides interactive selection)")
    parser.add_argument("--samples", type=int, help="Number of videos to record")
    parser.add_argument("--len", type=int, default=DEFAULT_SEQUENCE_LENGTH, help="Number of frames per video")
    parser.add_argument("--grace", type=int, default=DEFAULT_GRACE_PERIOD, help="Countdown seconds")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output folder")
    
    args = parser.parse_args()

    # Interactive action selection if not provided
    action = args.action
    if not action:
        actions = get_actions()
        print("\nAvailable Actions:")
        for idx, a in enumerate(actions):
            print(f"{idx+1}. {a}")
        try:
            choice = int(input(f"\nSelect action (1-{len(actions)}) or 0 for custom: "))
            if choice == 0:
                action = input("Enter custom action name: ").strip()
            else:
                action = actions[choice-1]
        except (ValueError, IndexError):
            print("Invalid selection. Exiting.")
            return

    # Interactive sample count if not provided
    samples = args.samples
    if samples is None:
        try:
            val = input(f"Enter number of samples (default 5): ").strip()
            samples = int(val) if val else 5
        except ValueError:
            print("Invalid number. Using default: 5")
            samples = 5

    if not action:
        print("No action specified. Exiting.")
        return

    record_raw_videos(action, samples, args.len, args.grace, args.outdir)

if __name__ == "__main__":
    main()
