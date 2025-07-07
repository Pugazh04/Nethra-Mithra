from gtts import gTTS

def text_to_speech(scene_description: str, filename: str = "scene_audio.mp3"):

    # Ensure sentence ends with a period
    if not scene_description.strip().endswith('.'):
        scene_description += '.'

    # Generate TTS audio
    tts = gTTS(text=scene_description, lang='en')
    tts.save(filename)

    print("Scene description audio generated and played.")

# Example usage
scene_text = "There is a clock and a laptop. The clock is on the left of the laptop"
#text_to_speech(scene_text)
