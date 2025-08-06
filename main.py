import os
import time
import sys
import logging
from logging.handlers import TimedRotatingFileHandler

from text_to_speech import describe_scene_audio
from face_recognition import capture_faces
from speech_to_text import wait_for_capture_command, wait_for_name
from object_detection import detect_objects, distance, are_objects_far_apart
from spatial_analysis import construct_relationships
from llm import describe_the_scene
from clean_text import clean_text_for_tts

log_file = "<add your path to nethra_mithra.log>"

handler = TimedRotatingFileHandler(
    log_file,
    when="midnight",    
    interval=1,
    backupCount=7,      
    encoding='utf-8',
    utc=False            
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[handler],
    force=True
)

logging.info("Modules imported")
logging.info('Welcome user !!!, I am Nethra Mithra, your personal visual assistant')
describe_scene_audio('Welcome user, I am Naythra Miththraa, your personal visual assistant')
logging.info("Say 'capture' or 'personalise' to start")
describe_scene_audio("Say capture or personalise to start")

while True:
    command = wait_for_capture_command()  
    if command == 'capture':
        logging.info("Capturing...")
        describe_scene_audio("Capturing")
        start_time = time.time()
        max_retries= 5
        retry_count= 0
       
        while retry_count < max_retries:
            objects = detect_objects()
            if not objects:
                retry_count+=1
                time.sleep(5)
                continue
           
            logging.info("Objects detected")
            det_obj= []
            object_counts= {}
            label_to_obj = {}
            for obj in objects:
                label = obj['label']
                label_to_obj.setdefault(label, []).append(obj)
            logging.info(f"Label to Object: {label_to_obj}")
            for label, objs in label_to_obj.items():
                object_counts[label] = len(objs)
                if len(objs) == 1:
                    det_obj.append(objs[0])
                elif are_objects_far_apart(objs):
                    det_obj.extend(objs)
                else:
                    det_obj.extend(objs[:2])  
           
            now = time.time()
            elapsed_time = now - start_time
           
            if (len(det_obj) >= 1):
                logging.info(f'Detected Objects : {det_obj}')
                logging.info(f"Objects Count : {object_counts}")
                rel = construct_relationships(det_obj)
                logging.info(f'Spatial Relationship : {rel}')
                desc = describe_the_scene(det_obj, rel, object_counts)
                if desc == 0:
                    logging.error("LLM model failed !!!")
                    describe_scene_audio("Sorry !!! We ran into a problem")
                    break
                logging.info(f'Description : {desc}')
                clean_text = clean_text_for_tts(desc)
                logging.info(f'Clean Text : {clean_text}')
                logging.info('Playing the audio...')
                describe_scene_audio(clean_text)
                break
           
            elif (elapsed_time >= 10):
                logging.error("Sorry !!! We ran into a problem")
                describe_scene_audio("Sorry !!! We ran into a problem")
                break
               
        else:
            logging.error("Max retries reached. No sufficient objects detected.")
            describe_scene_audio("Sorry!!! We couldn't find enough objects.")

   
    elif command == 'personalize':
        while True:
            logging.info('Please tell the name of the person')
            describe_scene_audio("Please tell the name of the person")
            name= wait_for_name()
            prompt= f"I heard {name}. Am I right ?"
            logging.info(prompt)
            describe_scene_audio(prompt)
            ans= wait_for_capture_command()
            if ans == 'yeah':
                logging.info("Thanks for confirming !!!")
                describe_scene_audio("Thanks for confirming !!!")
                res= capture_faces(name)
                if res == 1:
                    prompt= f"Successfully added {name}"
                    logging.info(prompt)
                    describe_scene_audio(prompt)
                else:
                    logging.error("Sorry !!! We ran into a problem")
                    describe_scene_audio("Sorry !!! We ran into a problem")
                break

    elif command == 'stop':
        logging.info("Thank you for using Nethra Mithraa !!!")
        describe_scene_audio("Thank you for using ,Naythra Mithraa")
        sys.stderr.close()
        break
   
    elif command == None:
        describe_scene_audio("Sorry !!! We ran into a problem")
       
    logging.info("Say 'capture' or 'personalize' to continue or 'stop' to end")
    describe_scene_audio("Say capture or personalize to continue or stop to end")


