# stage1_speech.py
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from langdetect import detect
import numpy as np

# -------------------------------
# Step 1: Record Audio
# -------------------------------
def record_audio(filename="speech.wav", duration=7, fs=16000):
    print(f"Recording for {duration} seconds. Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    # Convert to int16 for WAV format
    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    write(filename, fs, audio_int16)
    print(f"Recording saved as {filename}")
    return filename

# -------------------------------
# Step 2: Transcribe with Whisper
# -------------------------------
def transcribe_audio(audio_file):
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # base model is fast and accurate
    print("Transcribing audio...")
    result = model.transcribe(audio_file)
    text = result['text']
    print("Transcription complete.")
    return text

# -------------------------------
# Step 3: Language Detection
# -------------------------------
def detect_language(text):
    try:
        language = detect(text)
    except:
        language = "unknown"
    return language

# -------------------------------
# Step 4: Main Pipeline
# -------------------------------
def main():
    # 1️⃣ Record audio
    audio_file = record_audio(duration=7)

    # 2️⃣ Transcribe
    text_output = transcribe_audio(audio_file)
    print("\nTranscribed Text:")
    print(text_output)

    # 3️⃣ Detect language
    language = detect_language(text_output)
    print(f"\nDetected Language: {language}")
    if language == "en":
        print("✅ Input is English. Ready for Stage 2!")
    else:
        print("❌ Please speak in English.")

if __name__ == "__main__":
    main()
