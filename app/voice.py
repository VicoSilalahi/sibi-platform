import subprocess
import os

def bicara_piper(teks, model_path=None):
    """
    Speak text using Piper TTS.
    
    Args:
        teks (str): The text to speak.
        model_path (str, optional): Path to the .onnx model. Defaults to the standard news-medium model.
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        # Default model path relative to the project root
        model_path = os.path.join(current_dir, "piper", "id_ID-news_tts-medium.onnx")
    
    # Ensure we use the correct piper binary
    piper_bin = os.path.join(current_dir, "piper", "piper")
    if os.name == 'nt':  # Windows
        piper_bin += ".exe"
    
    # For Windows, we might need a different output method than aplay
    if os.name == 'nt':
        # On Windows, 'aplay' usually isn't there. Piper can output to a file or pipe.
        # This part might need further refinement for a real Windows env.
        command = f'echo {teks} | {piper_bin} --model {model_path} --output_raw | aplay -r 22050 -f S16_LE -t raw'
    else:
        command = f'echo "{teks}" | {piper_bin} --model {model_path} --output_raw | aplay -r 22050 -f S16_LE -t raw'
        
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    bicara_piper("tes 1, 2, 3 tes tes")
