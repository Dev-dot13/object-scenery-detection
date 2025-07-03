import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import os
import time

# Load Whisper model once globally
model = whisper.load_model("base")  # You can use "tiny" or "small" for faster performance

def record_audio(duration=5, output_path="outputs/input.wav"):
    """
    Records audio from the default microphone.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fs = 16000  # Whisper prefers 16kHz
    print(f"[INFO] Recording for {duration} seconds...")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(output_path, fs, recording)
    print(f"[INFO] Audio saved to {output_path}")
    return output_path

def transcribe_audio(audio_path):
    """
    Transcribes the recorded audio file using Whisper.
    """
    print("[INFO] Transcribing audio...")
    result = model.transcribe(audio_path)
    text = result["text"]
    print(f"[INFO] Transcription: {text}")
    return text
