import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Run prediction on image
results = model("test1.jpg")

# Show or save result
results[0].show()
results[0].save(filename="test1_output.jpg")

print(results)