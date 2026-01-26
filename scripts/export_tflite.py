import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.utils import convert_to_tflite

def export_all():
    print("Exporting Camera model to TF-Lite...")
    convert_to_tflite("camera_gru")
    
    print("Exporting Glove model to TF-Lite...")
    convert_to_tflite("glove_gru")

if __name__ == "__main__":
    export_all()
