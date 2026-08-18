"""
Vision API Test — Single Page
==============================
Page 10 (content page) render করে Vision API পাঠাই।
Bangla text আসছে কিনা verify করার জন্য।
"""
import io
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

import pypdfium2 as pdfium
from google.cloud import vision

PDF_PATH = r"data/raw/Secondary (BV)-2026_Class 9-10_Physics_compressed.pdf"
TEST_PAGE = 9  # 0-indexed → page 10 (likely a content page with text)
RENDER_DPI = 200  # good quality for OCR

print(f"[TEST] Rendering page {TEST_PAGE+1} at {RENDER_DPI} DPI...")

# Render PDF page to image
doc = pdfium.PdfDocument(PDF_PATH)
page = doc[TEST_PAGE]
scale = RENDER_DPI / 72  # pypdfium2 default is 72 DPI
bitmap = page.render(scale=scale, rotation=0)
pil_image = bitmap.to_pil()
doc.close()

# Save preview
pil_image.save("data/processed/test_page.png")
print(f"[SAVED] Preview: data/processed/test_page.png")
print(f"        Image size: {pil_image.size[0]}x{pil_image.size[1]} px")

# Convert to bytes
img_bytes = io.BytesIO()
pil_image.save(img_bytes, format="PNG")
img_bytes = img_bytes.getvalue()
print(f"        Image bytes: {len(img_bytes)/1024:.1f} KB")

# Send to Vision API
print("\n[API] Sending to Google Cloud Vision DOCUMENT_TEXT_DETECTION...")

client = vision.ImageAnnotatorClient()
image = vision.Image(content=img_bytes)
response = client.document_text_detection(image=image)

if response.error.message:
    print(f"[ERROR] {response.error.message}")
else:
    full_text = response.full_text_annotation.text
    print(f"[SUCCESS] Text received: {len(full_text)} chars")
    print(f"\n--- First 500 chars ---")
    print(full_text[:500])
    print("--- End ---")
    print(f"\nTotal words: {len(full_text.split())}")
