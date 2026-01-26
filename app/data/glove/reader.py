import numpy as np

class GloveReader:
    """Class to read raw sensor data from the glove hardware."""
    
    def __init__(self, port=None):
        self.port = port
        self.num_features = 22  # (5 flex + 6 MPU) * 2 hands

    def read_frame(self):
        """Read a single frame of sensor data from the hardware.
        
        Expected output format (22 features):
        [
            H1_F1, H1_F2, H1_F3, H1_F4, H1_F5,  # hand 1 flex
            H1_GX, H1_GY, H1_GZ, H1_AX, H1_AY, H1_AZ, # hand 1 mpu
            H2_F1, H2_F2, H2_F3, H2_F4, H2_F5,  # hand 2 flex
            H2_GX, H2_GY, H2_GZ, H2_AX, H2_AY, H2_AZ  # hand 2 mpu
        ]
        """
        # Placeholder for actual hardware reading (e.g., serial.readline())
        # The user should implement the actual hardware communication here.
        return np.zeros(self.num_features)

    def get_sequence(self, sequence_length=30):
        """Capture a sequence of frames."""
        sequence = []
        for _ in range(sequence_length):
            frame = self.read_frame()
            sequence.append(frame)
        return np.array(sequence)
