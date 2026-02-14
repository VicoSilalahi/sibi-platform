import numpy as np
import serial
import time

class GloveReader:
    """Class to read raw sensor data from the glove hardware via Serial."""
    
    def __init__(self, port='COM3', baudrate=115200, timeout=0.01):
        self.port = port
        self.baudrate = baudrate
        self.num_features = 22  # (5 flex + 6 MPU) * 2 hands
        self.ser = None
        self.last_frame = np.zeros(self.num_features)
        self.connected = False
        
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=timeout)
            time.sleep(2)  # Wait for serial connection to stabilize
            if self.ser.is_open:
                self.connected = True
                print(f"Connected to glove on {self.port}")
        except Exception as e:
            print(f"Warning: Could not connect to glove on {self.port}: {e}")

    def read_frame(self):
        """Read a single frame of sensor data from the hardware.
        
        Expected format: 22 comma-separated values ending with newline.
        """
        if self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    values = [float(x) for x in line.split(',')]
                    if len(values) == self.num_features:
                        self.last_frame = np.array(values)
                        return self.last_frame
            except Exception as e:
                pass # Silently fail to keep loop running
        
        # Return last valid frame or zeros if never read
        return self.last_frame

    def is_alive(self):
        """Check if we are actually receiving data."""
        if not self.connected or not self.ser or not self.ser.is_open:
            return False
            
        # Try to read a frame to see if data is flowing
        # Note: This might be slow if called frequently, but good for pre-checks
        if self.ser.in_waiting > 0:
            return True
        return False

    def get_sequence(self, sequence_length=30):
        """Capture a sequence of frames."""
        sequence = []
        for _ in range(sequence_length):
            frame = self.read_frame()
            sequence.append(frame)
        return np.array(sequence)

    def __del__(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
