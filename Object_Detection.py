import requests
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

model = YOLO("yolov8m.pt")
conf_threshold = 0.4

def detect_objects():
    try:
        esp32_url = "<add your ESP32 cam url>"
        response = requests.get(esp32_url, timeout=5)
        if response.status_code != 200:
            print("Failed to fetch image from ESP32-CAM")
            return []

        img_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)

        results = model(frame, verbose=False)
        detected_objects = []
        for result in results:
            for box in result.boxes:
                if box.conf[0] < conf_threshold:
                    continue
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_objects.append({
                    "label": class_name,
                    "bbox": (x1, y1, x2, y2)
                })
        annotated_frame = results[0].plot()
        cv2.imshow("ESP32-CAM YOLOv8 Detection", annotated_frame)
        cv2.imwrite("debug_frame.jpg", annotated_frame)
        cv2.waitKey(1)

        return detected_objects

    except Exception as e:
        print("Error:", e)
        return []
