from TTS.api import TTS
import os
import torch

# Load model once globally
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)

def speak_text(text, output_path="outputs/tts_output.wav"):
    """
    Synthesizes speech from the input text and saves it as a WAV file.
    Also plays the audio file using default player.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts.tts_to_file(text=text, file_path=output_path, speaker='p225')

    # Playback the audio using OS default player
    try:
        if os.name == 'nt':  # Windows
            os.system(f'start {output_path}')
        elif os.name == 'posix':  # Linux/macOS
            os.system(f'xdg-open {output_path}')
    except Exception as e:
        print(f"[WARNING] Could not play audio: {e}")
