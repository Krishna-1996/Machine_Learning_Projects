import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # DeepSpeech requires 16kHz
seconds = 30  # Duration of recording in seconds

print("Speak now...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

write("speech.wav", fs, audio)
print("Recording saved as speech.wav")
