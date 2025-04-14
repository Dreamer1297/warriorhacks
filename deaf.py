import sounddevice as sd
from scipy.io.wavfile import write
import os
from openai import OpenAI
from dotenv import load_dotenv
import assemblyai as aai
import subprocess

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")

transcriber = aai.Transcriber()

config = aai.TranscriptionConfig(
    auto_chapters=True,
    speaker_labels=True,
    sentiment_analysis=True,
    entity_detection=True,
    iab_categories=True,
    language_detection=True,
    language_confidence_threshold=0.4,
)

duration = 5
print("recording")
audio = sd.rec(duration*44100, samplerate=44100, channels=1)
sd.wait()
print("done")
write("recorded.wav", 44100, audio)

transcript = transcriber.transcribe(
    "./recorded.wav",
    config
)
print(transcript.text)

cmd = [
    "python3",
    "pyAudioAnalysis/audioAnalysis.py",
    "classifyFile",
    "-i", "./recorded.wav",
    "-m", "svm_rbf_sm"
]
result = subprocess.run(cmd, capture_output=True, text=True)
description = result.stdout
print(description)

orders = [
            {"role": "system", "content": f'''
                You are an AI model that generates a descriptions from transcribed text.
                Only output descriptions, nothing else. BE AS DESCRIPTIVE AS POSSIBLE USING ALL YOUR RESOURCES. BE EXTREMELEY RESOURCEFUL AND DESCRIPTIVE USING a classification of the speaker and transcribed text.

                given: Person saying 'How are you?' and classification = male, happy --> you should output: A person, probably male, is saying 'How are you?'
                given: Person saying 'Why did you cheat on me?' and classification = female, angry --> you should output: An angry female is saying "Why did you cheat on me?"
             
                Return only your sentences in that format. YOUR TONE: DO NOT DESCRIBE IT LIKE IT IS text. DESCRIBE IT LIKE IT IS THE HEARING OF A PERSON, 
             
                Combine the two resources into something that conveys the most information.
             
                Identify and predict emotion, etc.

                Here is the text: {transcript.text}
                Here is the classification of the speaker: {description}'''},

            {"role":"user", "content": f"Summarize what is happening. Your response should not take more than five seconds to say out loud."}
        ]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=orders,
).choices[0].message.content

print(response)