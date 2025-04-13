import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

capture = cv2.VideoCapture(0)
while capture.isOpened():
    
    ret, img = capture.read()
    if not ret: break
    height, width, channels = img.shape

    # process image
    results = model(img)

    # output confidence
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if (conf>0.7):
            id = int(box.cls[0])
            label = results[0].names[id]
            bbox = box.xyxy[0].tolist() #x1, y1, x2, y2
            #print(f"{label} @ {bbox} with {conf:.2f} confidence")

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("detection", img) #display processed image
    if cv2.waitKey(5) & 0xFF == 27: break