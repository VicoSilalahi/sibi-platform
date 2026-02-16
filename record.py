import cv2
import os
import time
import argparse
import socket
import csv
import sys
import numpy as np

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# --- Configuration (Hardcoded for Independence) ---
# You can modify these to suit your needs
# https://pulaubahasa.wordpress.com/vocab-builders/pbwl/ at root
# https://pkplk.kemendikdasmen.go.id/sibi/pencarian
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

class GloveReader:
    """Class to read raw sensor data from the glove hardware via Serial."""
    
    def __init__(self, port='COM3', baudrate=115200, timeout=0.01):
        self.port = port
        self.baudrate = baudrate
        self.num_features = 20  # (5 flex + 5 MPU) * 2 hands
        self.ser = None
        self.last_frame = [0.0] * self.num_features
        self.connected = False
        
        if not SERIAL_AVAILABLE:
            print("Warning: pyserial not installed. Glove mode disabled.")
            return

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=timeout)
            time.sleep(2)  # Wait for serial connection to stabilize
            if self.ser.is_open:
                self.connected = True
                print(f"Connected to glove on {self.port}")
        except Exception as e:
            print(f"Warning: Could not connect to glove on {self.port}: {e}")

    def read_frame(self):
        """Read a single frame of sensor data from the hardware."""
        if self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    values = [float(x) for x in line.split(',')]
                    if len(values) == self.num_features:
                        self.last_frame = values
                        return self.last_frame
            except Exception:
                pass
        return self.last_frame

    def is_alive(self):
        """Check if we are actually receiving data."""
        if not self.connected or not self.ser or not self.ser.is_open:
            return False
        return self.ser.in_waiting > 0

    def __del__(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

def record_data(mode, action, num_samples, sequence_length, grace_period, output_dir, glove_port='COM3'):
    """Record raw data (video and/or glove) for a specific action."""
    use_cam = mode in ['camera', 'both']
    use_glove = mode in ['glove', 'both']
    
    # Setup directories
    action_dir_cam = os.path.join(output_dir, 'videos', action) if use_cam else None
    action_dir_glove = os.path.join(output_dir, 'glove', action) if use_glove else None
    
    if action_dir_cam: os.makedirs(action_dir_cam, exist_ok=True)
    if action_dir_glove: os.makedirs(action_dir_glove, exist_ok=True)
    
    cap = None
    reader = None
    
    if use_cam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    if use_glove:
        reader = GloveReader(port=glove_port)
        if not reader.connected:
            print("Error: Glove connection failed. Check port or mode.")
            if use_cam: cap.release()
            return

    # Define codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    hostname = socket.gethostname().replace(" ", "_")
    
    print(f"\nMode: {mode.upper()}")
    print(f"Target Action: '{action}'")
    print(f"Samples: {num_samples}")
    print("-" * 30)
    
    for sample_num in range(num_samples):
        timestamp = int(time.time())
        filename_base = f"{hostname}_{timestamp}"
        
        video_path = os.path.join(action_dir_cam, f"{filename_base}.avi") if use_cam else None
        csv_path = os.path.join(action_dir_glove, f"{filename_base}.csv") if use_glove else None
        
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480)) if use_cam else None
        glove_data = []
        
        # Countdown
        print(f"\nRecording Sample {sample_num + 1}/{num_samples} in {grace_period}s...")
        for i in range(grace_period, 0, -1):
            start_wait = time.time()
            while time.time() - start_wait < 1:
                if use_cam:
                    ret, frame = cap.read()
                    if not ret: break
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"GET READY: {i}", (120, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                    cv2.imshow("SIBI Recorder", display_frame)
                    cv2.waitKey(1)
                else:
                    print(f"{i}...", end='\r')
                    time.sleep(0.1)
                    if time.time() - start_wait >= 1: break

        print(f"REC >>> Recording {action}...")
        
        start_time = time.time()
        for i in range(sequence_length):
            # Read Data
            frame = None
            if use_cam:
                ret, img = cap.read()
                if ret: 
                    frame = img
                    out.write(frame)
            
            if use_glove:
                glove_frame = reader.read_frame()
                glove_data.append(glove_frame)
            
            # Feedback
            if use_cam and frame is not None:
                display_frame = frame.copy()
                cv2.putText(display_frame, f"RECORDING: {action} | {i+1}/{sequence_length}", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("SIBI Recorder", display_frame)
            elif not use_cam:
                print(f"Recording: {i+1}/{sequence_length}", end='\r')
            
            # Maintain 30fps
            elapsed = time.time() - start_time
            expected = (i + 1) / 30.0
            if expected > elapsed:
                time.sleep(expected - elapsed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nCancelled.")
                if out: out.release()
                if cap: cap.release()
                cv2.destroyAllWindows()
                return
        
        if out: out.release()
        if glove_data:
            with open(csv_path, 'w', newline='') as f:
                writer_csv = csv.writer(f)
                writer_csv.writerows(glove_data)
        
        print(f"Saved: {filename_base}")
        time.sleep(0.5)
        
    if cap: cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Standalone SIBI Data Recorder")
    parser.add_argument("--mode", type=str, choices=['camera', 'glove', 'both'], help="Recording mode")
    parser.add_argument("--action", type=str, help="Action to record")
    parser.add_argument("--samples", type=int, default=5, help="Number of videos/logs to record")
    parser.add_argument("--len", type=int, default=60, help="Frames per sample")
    parser.add_argument("--grace", type=int, default=3, help="Countdown seconds")
    parser.add_argument("--outdir", type=str, default="raw_data", help="Output folder")
    parser.add_argument("--port", type=str, default="COM3", help="Glove serial port")
    
    args = parser.parse_args()

    # Interactive mode selection
    mode = args.mode
    if not mode:
        print("\nSelect Mode:")
        print("1. Camera Only")
        print("2. Glove Only")
        print("3. Both (Synchronized)")
        try:
            m_choice = int(input("Choice (1-3): "))
            mode = ['camera', 'glove', 'both'][m_choice-1]
        except (ValueError, IndexError):
            print("Invalid. Defaulting to camera.")
            mode = 'camera'

    # Interactive action selection
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

    if not action:
        print("No action specified. Exiting.")
        return

    record_data(mode, action, args.samples, args.len, args.grace, args.outdir, args.port)

if __name__ == "__main__":
    main()
