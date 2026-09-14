import os
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()
text = "হ্যালো ভাইয়া! কেমন আছো? আমি তোমার ফিজিক্স টিউটর। চলো, আজকে নিউটনের সূত্র নিয়ে মজা করে শিখি!"

desktop_path = r"C:\Users\Home\Desktop"

chirp_voices = [
    "bn-IN-Chirp3-HD-Charon",
    "bn-IN-Chirp3-HD-Alnilam",
]

for voice_name in chirp_voices:
    voice = texttospeech.VoiceSelectionParams(
        language_code="bn-IN",
        name=voice_name
    )
    
    # Cordial tone: slightly slower rate. No pitch for Chirp.
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.9, 
    )

    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    try:
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        file_path = os.path.join(desktop_path, f"tutor_sample_{voice_name}.mp3")
        with open(file_path, "wb") as out:
            out.write(response.audio_content)
        print(f"Saved {file_path}")
    except Exception as e:
        print(f"Failed {voice_name}: {e}")
