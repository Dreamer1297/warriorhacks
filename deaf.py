import sounddevice as sd
from scipy.io.wavfile import write
import os
from openai import OpenAI
from dotenv import load_dotenv
import assemblyai as aai

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")

transcriber = aai.Transcriber()

duration = 5
print("recording")
audio = sd.rec(duration*44100, samplerate=44100, channels=1)
sd.wait()
print("done")
write("recorded.wav", 44100, audio)

transcript = transcriber.transcribe(
    "./recorded.wav",
    #auto_chapters=True,
    #speaker_labels=True,
    #sentiment_analysis=True,
    #entity_detection=True,
    #iab_categories=True
)

print("text:", transcript.text)

orders = [
            {"role": "system", "content": f'''
                You are an AI model that generates a descriptions from transcribed text.
                Identify emotion, .
                Only output descriptions, nothing else. BE AS DESCRIPTIVE AS POSSIBLE USING ALL YOUR RESOURCES. BE EXTREMELEY RESOURCEFUL AND DESCRIPTIVE USING THE ORIGINAL IMAGE, THE BOUNDING BOX DATA OF THE OBJECTS IN THAT IMAGE, AND THE OBJECTS IN THAT IMAGE.

                given: audio of slight jazz music and people talking --> you should output: There is jazz music playing, and people are chatting.
                given: audio of many cars honking --> you should output: Many cars are honking.
                given: audio of a person saying 'Hello! How are you doing?' --> you should output: A person is saying 'Hello! How are you doing?'
             
                Return only your sentences in that format. Also please, when talking about the location, DO NOT DESCRIBE THE LOCATION. ONLY SAY THE LOCATION. BE ONLY DESCRIPTIVE ABOUT THE SUBJECTS.
             
                YOUR TONE: DO NOT DESCRIBE IT LIKE IT IS text. DESCRIBE IT LIKE IT IS THE HEARING OF A PERSON, 
             
                Here is the audio file: {transcript.text}'''},

            {"role":"user", "content": f"Summarize what is happening. Your response should not take more than five seconds to say out loud."}
        ]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=orders,
).choices[0].message.content

print(response)