import cv2
import time
from ultralytics import YOLO
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

model = YOLO("yolov8n.pt")
client = OpenAI(api_key=os.getenv("API_KEY"))

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FPS, 1)

prev = time.time()

while capture.isOpened():
    ret, img = capture.read()
    if not ret:
        break

    height, width, channels = img.shape
    results = model(img)

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf > 0.65:
            id = int(box.cls[0])
            label = results[0].names[id]
            bbox = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Only process once every 5 seconds
    if time.time() - prev > 5:
        prev = time.time()

        messages = [
            {"role": "system", "content": f'''
                You are an AI model that generates a sentence from object data, designed for helping the visually impaired understand object interactions.
                You are given a list of detected objects and their bounding boxes from a list
                Infer likely relationships between the objects in the form of a sentence.
                Only output sentences — nothing else. Do NOT include any reasoning, or speculation. Only use objects that are likely present based on the list. No extra words. No paragraphs. No explanations.

                Example Output Format:
                person, dog, bbox data --> A person is walking a dog.
                car, bus, bbox data --> A car is behind a bus.

                Return only your sentences in that format.
            '''},
            {"role":"system", "content": f"Here is the list: {results[0]}"},
            {"role":"user", "content": f"Summarize what is happening."}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            ).choices[0].message.content

            print(response)

            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="nova",
                input=response,
            ) as speech:
                speech.stream_to_file("speech.mp3")

            # Optional: play it automatically
            os.system("afplay speech.mp3")  # macOS only

        except Exception as e:
            print(f"Error: {e}")

    # Display frame (optional)
    cv2.imshow("Scene", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()