import os
import vertexai
from config import Settings
from services.gemini_service import GeminiService

settings = Settings()
vertexai.init(project=settings.gcp_project, location=settings.gcp_location)
gemini = GeminiService(settings)

image_path = r"C:\Users\Home\.gemini\antigravity\brain\efddd743-12c8-4163-8256-ac9fb13706ca\physics_problem_test_1788371280725.jpg"

print("Reading image...")
with open(image_path, "rb") as f:
    image_bytes = f.read()

print("Sending image to AI...")
result = gemini.analyze_image(image_bytes, "ভাইয়া এই অংকটা কি ঠিক আছে?")

print("-" * 50)
print("AI Understanding:")
print("-" * 50)
print(result)
print("-" * 50)
