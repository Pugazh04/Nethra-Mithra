import subprocess
import logging

def describe_scene_audio(text):
    try:
        piper_process = subprocess.Popen(
            ["piper", "--model", "en_US-amy-medium.onnx", "--output_file", "-", "--output_raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        aplay_process = subprocess.Popen(
            ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
            stdin=piper_process.stdout
        )

        piper_process.stdin.write(text.encode("utf-8"))
        piper_process.stdin.close()

        piper_process.stdout.close()
        aplay_process.communicate()

    except Exception as e:
        logging.error(f"Failed to speak using Piper TTS: {e}")


