import sounddevice as sd
from scipy.io.wavfile import write
import os
from openai import OpenAI

duration = 5  # seconds
fs = 44100  # Sample rate

print("Recording...")
audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
sd.wait()  # Wait until recording is finished
print("Done!")

write("recorded.wav", fs, audio)  # Save as WAV file