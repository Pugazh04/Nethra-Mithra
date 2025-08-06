import cv2
import os
import numpy as np
import pickle
from datetime import datetime
import requests
import logging
import time

from text_to_speech import describe_scene_audio

BASE_DIR = "face_data"
FACES_DIR = os.path.join(BASE_DIR, "faces")
MODEL_PATH = os.path.join(BASE_DIR, "face_recognizer.xml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.pkl")
TIMEOUT = 180
NUM_IMAGES = 50
CONFIDENCE_THRESHOLD = 90

os.makedirs(FACES_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def capture_faces(person_name):
    person_dir = os.path.join(FACES_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    count = 0
    logging.info(f"Capturing 50 face images for '{person_name}'...")
    start_time = time.time()
    max_retries= 5
    retry_count= 0
    while (count < NUM_IMAGES):
        if time.time() - start_time > TIMEOUT:
            logging.error("Timeout reached while capturing faces. Exiting...")
            break
        try:
            esp32_url = "<add your ESP32-CAM url here>"
            logging.info("Fetching image...")
            response = requests.get(esp32_url, timeout=5)
            logging.info(f"Response Status: {response.status_code}")
            response.raise_for_status()  
            if response.status_code == 200:
                logging.info("Image successfully fetched")
            else:
                logging.error(f"Failed to fetch image: HTTP Status Code {response.status_code}")
                continue
            img_array = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                logging.error("Frame could not be decoded")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
            if len(faces) == 0:
                logging.warning("No face detected in the frame.")
                describe_scene_audio("No face detected. Please face the camera properly.")
                if retry_count < max_retries:
                    retry_count+=1
                    logging.info(f"Retrying... Attempt {retry_count}/{max_retries}")
                else:
                    break
                time.sleep(5)
                continue
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                img_path = os.path.join(person_dir, f"{count}.jpg")
                cv2.imwrite(img_path, face_img)
                count += 1
                logging.info(f"Captured: {count}/50")
                break  
        except Exception as e:
            logging.error(f"ESP32 Capture Failed: {e}")
            describe_scene_audio("Camera couldn't fetch image. Please check the connection.")
            if retry_count < max_retries:
                retry_count+=1
                logging.info(f"Retrying... Attempt {retry_count}/{max_retries}")
            else:
                break
            time.sleep(10)    

    if count >= NUM_IMAGES:
        logging.info(f"Finished capturing images for {person_name}")
        op= train_lbph_model()
        if op == 1:
            return 1
        else:
            return 0
    else:
        logging.warning(f"Only captured {count} faces. Training skipped.")
        return 0

def train_lbph_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    current_id = 0
    label_ids = {}
    x_train = []
    y_labels = []

    for root, dirs, files in os.walk(FACES_DIR):
        for person in dirs:
            person_path = os.path.join(FACES_DIR, person)
            label = label_ids.setdefault(person, current_id)
            if label == current_id:
                current_id += 1

            for file in os.listdir(person_path):
                path = os.path.join(person_path, file)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    logging.warning(f"LBPH Trainer ---> Skipping unreadable image: {path}")
                    continue
                faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
                for (x, y, w, h) in faces:
                    face = image[y:y+h, x:x+w]
                    x_train.append(face)
                    y_labels.append(label)
    if not x_train:
        logging.error("LBPH Trainer ---> No training data found.")
        return 0
    logging.info(f"Collected {len(x_train)} face samples for training.")
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_ids, f)

    logging.info("LBPH model trained and saved.")
    return 1

def recognize_faces_in_image(image):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        logging.warning("Model and/or label file not found.")
        return ["Unknown Person"]

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "rb") as f:
        labels = {v: k for k, v in pickle.load(f).items()}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
    recognized_names = []
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi)
        if conf < CONFIDENCE_THRESHOLD:
            logging.debug(f"Prediction: id={id_}, conf={conf}")
            name = labels.get(id_, "Unknown Person")
        else:
            name= "Unknown Person"
        recognized_names.append(name)
    return recognized_names


