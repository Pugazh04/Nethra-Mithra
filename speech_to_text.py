import os
import sys
import json
import sounddevice as sd
import logging

def suppress_stderr():
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

suppress_stderr()

from vosk import Model, KaldiRecognizer

def wait_for_capture_command():
    try:
        model = Model("<add your path to Vosk model here>"")
    except Exception as e:
        logging.error(f"Error loading Vosk model: {e}")
        return None
    commands= ['capture', 'personalize', 'yeah', 'no', 'stop']
    recognizer = KaldiRecognizer(model, 16000, json.dumps(commands))
   
    try:
        with sd.InputStream(samplerate=16000, channels=1, dtype='int16') as stream:
            while True:
                try:
                    data = stream.read(4000)[0]
                    if recognizer.AcceptWaveform(data.tobytes()):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").lower().strip()
                        logging.info(f"Speech Detected: {text}")
                        if text in commands:
                            return text
                except Exception as stream_err:
                    logging.error(f"Error during speech recognition: {stream_err}")
    except Exception as e:
        logging.error(f"Failed to open audio input stream: {e}")
        return None
   
def wait_for_name():
    try:
        model = Model("<add your path to Vosk model here>")
    except Exception as e:
        logging.error(f"Error loading Vosk model: {e}")
        return None
    recognizer = KaldiRecognizer(model, 16000)

    try:
        with sd.InputStream(samplerate=16000, channels=1, dtype='int16') as stream:
            while True:
                try:
                    data = stream.read(4000)[0]
                    if recognizer.AcceptWaveform(data.tobytes()):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip().title()
                        logging.info(f"Name Detected: {text}")
                        return text
                except Exception as stream_err:
                    logging.error(f"Error during name capture: {stream_err}")
    except Exception as e:
        logging.error(f"Failed to open audio input stream: {e}")
        return None

