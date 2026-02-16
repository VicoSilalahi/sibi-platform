import subprocess

def bicara_piper(teks):
    # Ganti path model dengan file .onnx yang sudah kamu download
    model = "./piper/id_ID-news_tts-medium.onnx" 
    command = f'echo "{teks}" | ./piper/piper --model {model} --output_raw | aplay -r 22050 -f S16_LE -t raw'
    subprocess.run(command, shell=True)

bicara_piper("tes 1, 2, 3 tes tes")