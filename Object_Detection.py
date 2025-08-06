import requests
import os
from datetime import datetime
import time
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import math

from text_to_speech import describe_scene_audio
from face_recognition import recognize_faces_in_image

model = YOLO("yolov8s.pt")
model.fuse()
_ = model.predict(np.zeros((640, 640, 3), dtype=np.uint8))

def distance(obj1, obj2):
    x1 = (obj1['bbox'][0] + obj1['bbox'][2]) / 2
    y1 = (obj1['bbox'][1] + obj1['bbox'][3]) / 2
    x2 = (obj2['bbox'][0] + obj2['bbox'][2]) / 2
    y2 = (obj2['bbox'][1] + obj2['bbox'][3]) / 2
    return math.hypot(x2 - x1, y2 - y1)

def are_objects_far_apart(objects, threshold=100):
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if distance(objects[i], objects[j]) > threshold:
                return True
    return False

def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

def detect_objects():
    try:
        esp32_url = "<add your ESP32-CAM url here>"
        logging.info("Fetching image...")
        response = requests.get(esp32_url, timeout=5)
        logging.info(f"Response Status: {response.status_code}")
        response.raise_for_status()  
        if response.status_code == 200:
            logging.info("Image successfully fetched")
        img_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            logging.error("Frame could not be decoded")
            describe_scene_audio("Camera couldn't fetch image. Please check the connection.")
            return []  
        results = model.predict(frame, verbose=False, conf=0.6)
        if not results or len(results) == 0:
            logging.warning("No objects detected")
            describe_scene_audio("No objects found. Try capturing a clearer image")
            return []  
        detected_objects = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                new_bbox = (x1, y1, x2, y2)
                should_add = True
                for existing_obj in detected_objects:
                    iou = calculate_iou(existing_obj["bbox"], new_bbox)
                    if iou > 0.4:
                        if existing_obj["label"] == class_name:
                            should_add = False
                            break
                        logging.warning(f"Overlap conflict: {class_name} vs {existing_obj['label']}, IoU={iou:.2f}")
                        should_add = False
                        break
                if should_add:
                    detected_objects.append({
                        "label": class_name,
                        "bbox": new_bbox,
                        "name": class_name
                    })

        annotated_frame = results[0].plot()
        save_debug_frame(annotated_frame)
       
        if any(obj["label"] == "person" for obj in detected_objects):
            faces= recognize_faces_in_image(frame)
            face_idx = 0
            for obj in detected_objects:
                if obj["label"] == "person":
                    if face_idx < len(faces):
                        obj["label"] = faces[face_idx]
                        face_idx += 1
        return detected_objects

    except Exception as e:
        logging.error(f"Failed to fetch image: HTTP Status Code {response.status_code}")
        describe_scene_audio("Camera couldn't fetch image. Please check the connection.")
        return []

def save_debug_frame(annotated_frame):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f")
        filename = f"debug/debug_frame_{timestamp}.jpg"
        cv2.imwrite(filename, annotated_frame)
    except Exception as e:
        logging.error(f"Error saving frame: {e}")
