import cv2
import time
from ultralytics import YOLO
from collections import Counter
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
        if conf > 0.6:
            id = int(box.cls[0])
            label = results[0].names[id]
            bbox = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # once every 5 seconds
    if time.time() - prev > 5:
        prev = time.time()

        labels = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
        things = Counter(labels)

        orders = [
            {"role": "system", "content": f'''
                You are an AI model that generates a sentence from the YOLO object detection data outputted with the command model(img) and the original image,designed for helping the visually impaired understand object interactions.
                You are given a list of detected objects.
                Infer likely relationships between the objects and what is happening
                Only output descriptions, nothing else. BE AS DESCRIPTIVE AS POSSIBLE USING ALL YOUR RESOURCES. BE EXTREMELEY RESOURCEFUL AND DESCRIPTIVE USING THE ORIGINAL IMAGE, THE BOUNDING BOX DATA OF THE OBJECTS IN THAT IMAGE, AND THE OBJECTS IN THAT IMAGE.

                given: person, dog, bbox data, img, objects --> you should output: A person is walking a dog, and the dog has brown fur. The person has a smile on his face.
                given: car, bus, person, person, person, bbox data, img, objects --> you should output: A red car is behind a large bus, and three people are walking around. One person has a red hat.
                Return only your sentences in that format. Also please, when talking about the location, DO NOT DESCRIBE THE LOCATION. ONLY SAY THE LOCATION. BE ONLY DESCRIPTIVE ABOUT THE SUBJECTS.
             
                Take into account the size of the bounding boxes, like a smaller bbox might mean an object is farther away or just smaller. Use all these resources (like the image itself) to accurately describe the image.
                The data is an array with the value of model(img)[0].boxes. Here is the data: {results[0].boxes}
                This is a list of all the objects in the image: {things}
                And here is the original image: {img}'''},
            {"role":"user", "content": f"Summarize what is happening. Your response should not take more than five seconds to say out loud."}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=orders,
            ).choices[0].message.content

            print(response)

            # speak text
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="nova",
                input=response,
            ) as speech:
                speech.stream_to_file("speech.mp3")

            # Optional: play it automatically
            os.system("afplay speech.mp3")

        except Exception as e:
            print(f"Error: {e}")

    print()
    print()
    print()
    print()
    print("             results: ")
    print()
    print()
    print()
    print()
    print(results[0].boxes)

    # Display frame (optional)
    cv2.imshow("Scene", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()