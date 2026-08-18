"""
Step 1 (v2): Full PDF Extraction via Google Cloud Vision API
=============================================================
Strategy:
  - pypdfium2 → render each page as PNG at 200 DPI
  - Google Cloud Vision DOCUMENT_TEXT_DETECTION → proper Bangla Unicode OCR
  - Incremental write + progress tracking → crash-safe, resumable
  - Cost: ~$0.55 for 366 pages (well within free credit)

Prerequisites (already done):
  - google-cloud-vision installed
  - ADC credentials set with quota_project_id
  - Vision API enabled on project-3e580a5b-256c-4c6d-a0a
"""

import io
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

import pypdfium2 as pdfium
from google.cloud import vision

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
PDF_PATH    = Path("data/raw/Secondary (BV)-2026_Class 9-10_Physics_compressed.pdf")
OUTPUT_MD   = Path("data/processed/extracted_book.md")
OUTPUT_JSON = Path("data/processed/extracted_document.json")
PROGRESS_F  = Path("data/processed/vision_progress.json")

RENDER_DPI  = 200   # good quality for OCR, manageable file size
RETRY_MAX   = 3     # retries per page on API error
RETRY_DELAY = 5     # seconds between retries

OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------
# Progress tracking (crash-safe resume)
# ----------------------------------------------------------------
def load_progress() -> dict:
    if PROGRESS_F.exists():
        with open(PROGRESS_F, encoding="utf-8") as f:
            return json.load(f)
    return {"done_pages": [], "failed_pages": [], "total_chars": 0}


def save_progress(p: dict):
    with open(PROGRESS_F, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2)


# ----------------------------------------------------------------
# Render one PDF page to PNG bytes
# ----------------------------------------------------------------
def render_page(doc: pdfium.PdfDocument, page_idx: int) -> bytes:
    page   = doc[page_idx]
    scale  = RENDER_DPI / 72
    bitmap = page.render(scale=scale, rotation=0)
    img    = bitmap.to_pil()
    buf    = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------
# Call Vision API with retry
# ----------------------------------------------------------------
def ocr_page(client: vision.ImageAnnotatorClient, img_bytes: bytes) -> str:
    image    = vision.Image(content=img_bytes)
    for attempt in range(RETRY_MAX):
        try:
            response = client.document_text_detection(image=image)
            if response.error.message:
                raise RuntimeError(response.error.message)
            return response.full_text_annotation.text or ""
        except Exception as e:
            if attempt < RETRY_MAX - 1:
                print(f" [retry {attempt+1}]", end="", flush=True)
                time.sleep(RETRY_DELAY)
            else:
                raise e


# ----------------------------------------------------------------
# Format page text as clean markdown section
# ----------------------------------------------------------------
def format_page_md(page_num: int, text: str) -> str:
    # Normalise line endings, strip trailing whitespace per line
    lines = [l.rstrip() for l in text.replace("\r\n", "\n").split("\n")]
    clean = "\n".join(lines).strip()
    return f"\n\n<!-- Page {page_num} -->\n{clean}"


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    if not PDF_PATH.exists():
        print(f"[ERROR] PDF not found: {PDF_PATH}")
        sys.exit(1)

    doc         = pdfium.PdfDocument(str(PDF_PATH))
    total_pages = len(doc)
    client      = vision.ImageAnnotatorClient()

    progress    = load_progress()
    done        = set(progress["done_pages"])
    failed      = progress["failed_pages"]
    total_chars = progress["total_chars"]

    if done:
        print(f"[RESUME] {len(done)} pages already done — continuing from where we left off")
    else:
        # Fresh start — clear output file
        OUTPUT_MD.write_text("", encoding="utf-8")

    print(f"\n[PDF]  {PDF_PATH.name}  ({total_pages} pages)")
    print(f"[API]  Google Cloud Vision — DOCUMENT_TEXT_DETECTION")
    print(f"[DPI]  {RENDER_DPI}  |  Estimated cost: ${total_pages * 0.0015:.2f}\n")

    for page_idx in range(total_pages):
        page_num = page_idx + 1

        if page_idx in done:
            pct = page_num / total_pages * 100
            print(f"[{pct:5.1f}%] Page {page_num:3d}/{total_pages}  SKIP")
            continue

        pct = page_num / total_pages * 100
        print(f"[{pct:5.1f}%] Page {page_num:3d}/{total_pages} ...", end=" ", flush=True)

        try:
            img_bytes = render_page(doc, page_idx)
            text      = ocr_page(client, img_bytes)
            chars     = len(text)
            total_chars += chars

            # Append immediately to output file
            with open(OUTPUT_MD, "a", encoding="utf-8") as f:
                f.write(format_page_md(page_num, text))

            done.add(page_idx)
            progress["done_pages"]  = list(done)
            progress["total_chars"] = total_chars
            save_progress(progress)

            print(f"OK  ({chars:4d} chars)")

        except Exception as e:
            print(f"FAILED: {e}")
            failed.append({"page": page_num, "error": str(e)})
            progress["failed_pages"] = failed
            save_progress(progress)

        # Small delay to be respectful to API rate limits
        time.sleep(0.1)

    doc.close()

    # ----------------------------------------------------------------
    # Save index JSON
    # ----------------------------------------------------------------
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "source_pdf"  : str(PDF_PATH),
            "total_pages" : total_pages,
            "completed"   : len(done),
            "failed"      : failed,
            "total_chars" : total_chars,
            "markdown_file": str(OUTPUT_MD),
        }, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    md_kb = OUTPUT_MD.stat().st_size / 1024
    print(f"\n{'='*52}")
    print(f"[DONE]  {len(done)}/{total_pages} pages completed")
    if failed:
        print(f"[WARN]  {len(failed)} pages failed: {[f['page'] for f in failed]}")
    print(f"[SIZE]  {md_kb:.0f} KB  |  {total_chars:,} total chars")
    print(f"[SAVED] {OUTPUT_MD}")
    print(f"\nNext: python scripts/2_chunk.py")

    # Delete progress file on full success
    if len(done) == total_pages:
        PROGRESS_F.unlink(missing_ok=True)
        print("[INFO]  Progress file cleaned up.")


if __name__ == "__main__":
    main()
