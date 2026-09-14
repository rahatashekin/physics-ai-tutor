import os
import wave
from google import genai
from google.genai import types
from config import Settings

settings = Settings()
client = genai.Client(vertexai=True, project=settings.gcp_project, location=settings.gcp_location)

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

voices = ["Charon", "Alnilam"]
desktop_path = r"C:\Users\Home\Desktop"
text = "হ্যালো ভাইয়া! কেমন আছো? আমি তোমার ফিজিক্স টিউটর। চলো, আজকে নিউটনের সূত্র নিয়ে মজা করে শিখি!"

for voice_name in voices:
    prompt = f"Speak the following Bengali text in a very cordial, warm, friendly, and enthusiastic male tone, like an encouraging older brother: {text}"
    
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-tts-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                speech_config=types.SpeechConfig(
                    language_code="bn-IN",
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                )
            )
        )
        
        data = response.candidates[0].content.parts[0].inline_data.data
        file_path = os.path.join(desktop_path, f"gemini_tts_{voice_name}.wav")
        wave_file(file_path, data)
        print(f"Saved {file_path}")
        
    except Exception as e:
        print(f"Failed {voice_name}: {e}")
