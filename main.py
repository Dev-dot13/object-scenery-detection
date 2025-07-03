from utils.stt import record_audio, transcribe_audio
from utils.intent import is_describe_intent
from utils.camera import capture_image
from utils.detect import detect_objects
from utils.caption import generate_caption
from utils.tts import speak_text
import time

def main():
    print("\n🎤 Speak your command (e.g., 'what do you see?')\n")

    # 1. Record and transcribe command
    audio_file = record_audio(duration=5)
    user_command = transcribe_audio(audio_file).lower()

    if is_describe_intent(user_command):
        print("\n📸 Capturing image from camera...\n")
        image_path = capture_image()

        print("\n🔍 Detecting objects in image...\n")
        objects = detect_objects(image_path)

        print("\n📝 Generating caption...\n")
        caption = generate_caption(image_path)

        # Combine caption + objects into a final sentence
        if objects:
            object_str = ", ".join(objects)
            final_response = f"It looks like: {caption}. Overall things that I see here are: {object_str}."
        else:
            final_response = f"It looks like: {caption}."

        print(f"\n🗣️ Responding: {final_response}\n")
        speak_text(final_response)

    else:
        print("\n🤖 I'm not sure how to help with that. Try asking me things like: 'What do you see?', or 'Describe your surroundings'\n")
        speak_text("I'm not sure how to help with that. Try asking me things like: 'What do you see?', or 'Describe your surroundings'")

if __name__ == "__main__":
    main()
