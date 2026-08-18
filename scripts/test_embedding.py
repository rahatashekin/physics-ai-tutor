"""Test Vertex AI embedding with ADC credentials"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from google import genai

# Vertex AI mode: API key লাগবে না, ADC credentials use করবে
client = genai.Client(
    vertexai=True,
    project="project-3e580a5b-256c-4c6d-a0a",
    location="us-central1",
)

test_texts = [
    "ত্বরণ কাকে বলে?",
    "Acceleration is the rate of change of velocity.",
    "v = u + at",
]

print("Testing Vertex AI embeddings...")
result = client.models.embed_content(
    model="text-embedding-004",
    contents=test_texts,
)

for i, emb in enumerate(result.embeddings):
    vec = emb.values
    print(f"[{i}] text: {repr(test_texts[i][:40])}")
    print(f"     dim: {len(vec)}, first 3: {[round(v,4) for v in vec[:3]]}")
    print()

print("SUCCESS! Vertex AI embeddings working.")
