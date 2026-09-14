import sys
import vertexai
from google import genai as google_genai
from config import Settings

settings = Settings()
vertexai.init(project=settings.gcp_project, location=settings.gcp_location)
client = google_genai.Client(vertexai=True)

try:
    response = client.models.generate_content(
        model="publishers/google/models/gemini-2.5-flash-image",
        contents="A simple pendulum"
    )
    print("RESPONSE DUMP:")
    print(response)
    print("\nPARTS:")
    for i, part in enumerate(response.candidates[0].content.parts):
        print(f"Part {i}:")
        print(f"  inline_data: {bool(part.inline_data)}")
        if part.inline_data:
            print(f"  mime_type: {part.inline_data.mime_type}")
            print(f"  data len: {len(part.inline_data.data)}")
        print(f"  text: {bool(part.text)}")
except Exception as e:
    print(f"Error: {e}")
