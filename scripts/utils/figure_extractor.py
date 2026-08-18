"""
scripts/utils/figure_extractor.py
===================================
PURPOSE:
    Physics book PDF থেকে সব figure (ছবি/চিত্র) crop করে save করা।

HOW IT WORKS:
    Step 1: Docling দিয়ে পুরো PDF এর layout analyze করে
            (do_ocr=False — text দরকার নেই, শুধু figure বাক্স দরকার)
    Step 2: প্রতিটা PICTURE item এর bounding box নেওয়া
    Step 3: pypdfium2 দিয়ে সেই অংশ crop করে PNG save করা
    Step 4: figure_locations.json এ সব info রাখা
            (পরে 2_chunk.py এ caption + figure_id match করবে)

COORDINATE SYSTEM NOTE:
    Docling bbox: PDF coordinate system (origin = bottom-left, y বাড়ে উপরে)
        l = left edge (points from left)
        t = top edge (points from bottom — higher number = higher on page)
        r = right edge
        b = bottom edge (points from bottom — lower number = lower on page)
    
    pypdfium2 crop: same PDF coordinate system
        crop=(left, bottom, right, top) — same as (bbox.l, bbox.b, bbox.r, bbox.t)

CRASH SAFETY:
    Progress file রাখে — যদি crash করে, restart করলে already-done pages skip করবে

USAGE:
    # Environment variable দিতে হবে (Windows এ MSVC না থাকলে)
    $env:TORCHDYNAMO_DISABLE="1"
    python scripts/utils/figure_extractor.py

    # অথবা একসাথে:
    $env:TORCHDYNAMO_DISABLE="1"; python scripts/utils/figure_extractor.py
"""

import io
import json
import os
import sys
from pathlib import Path

# Windows terminal UTF-8 fix
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
PDF_PATH        = Path("data/raw/Secondary (BV)-2026_Class 9-10_Physics_compressed.pdf")
FIGURES_DIR     = Path("data/processed/figures")
LOCATIONS_JSON  = Path("data/processed/figure_locations.json")
PROGRESS_FILE   = Path("data/processed/figure_extraction_progress.json")
PAGE_MAP_CACHE  = Path("data/processed/page_map_cache.json")

CROP_DPI        = 200   # figure image quality (200 DPI = good quality, manageable size)
MIN_FIGURE_AREA = 2000  # minimum area in PDF points² — tiny misdetections বাদ দেব
                         # (এর চেয়ে ছোট "figure" আসলে icon/bullet হওয়ার সম্ভাবনা বেশি)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Page map load
# ─────────────────────────────────────────────
def load_page_map() -> dict[int, int]:
    """PDF page → book page mapping (Step 1 এ তৈরি হয়েছে)"""
    if not PAGE_MAP_CACHE.exists():
        raise FileNotFoundError(
            "page_map_cache.json not found! "
            "Please run: python scripts/utils/book_page_mapper.py first"
        )
    with open(PAGE_MAP_CACHE, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


# ─────────────────────────────────────────────
# Progress tracking (crash-safe)
# ─────────────────────────────────────────────
def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"done_pages": [], "figures_found": []}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# Figure area calculation (tiny figures reject করব)
# ─────────────────────────────────────────────
def bbox_area(bbox) -> float:
    """Docling bbox এর area calculate করা (PDF points squared)"""
    width  = abs(bbox.r - bbox.l)
    height = abs(bbox.t - bbox.b)
    return width * height


# ─────────────────────────────────────────────
# Crop একটা figure এবং PNG save করা
# ─────────────────────────────────────────────
def crop_and_save_figure(
    pdf_doc,              # pypdfium2 PdfDocument
    pdf_page_idx: int,    # 0-based page index
    bbox,                 # Docling BoundingBox
    output_path: Path,
) -> bool:
    """
    PDF এর একটা page এর নির্দিষ্ট অংশ crop করে PNG save করে।

    Coordinate System:
        Docling bbox → PDF coords (origin = bottom-left, y বাড়ে উপরে)
            l = left, r = right (x from left)
            b = bottom, t = top (y from bottom — t > b)

        pypdfium2 render → screen coords (origin = top-left, y বাড়ে নিচে)
            তাই y-axis flip করতে হবে:
            pixel_y = (page_height - pdf_y) * scale

    Strategy:
        1. Full page render করব (PIL image)
        2. PDF coords → pixel coords convert করব (y-flip সহ)
        3. PIL দিয়ে crop করব
        → এটা pypdfium2 এর built-in crop parameter এর চেয়ে বেশি reliable

    Returns:
        True = সফল, False = ব্যর্থ
    """
    try:
        page       = pdf_doc[pdf_page_idx]
        page_w_pts = page.get_width()   # page width in PDF points
        page_h_pts = page.get_height()  # page height in PDF points

        scale  = CROP_DPI / 72   # 72 points/inch → pixel scale
        bitmap = page.render(scale=scale, rotation=0)
        img    = bitmap.to_pil()

        # PDF coords → pixel coords (y-axis flip)
        # PDF: y=0 at bottom, y increases upward
        # PIL: y=0 at top,    y increases downward
        px_left   = int(bbox.l * scale)
        px_right  = int(bbox.r * scale)
        px_top    = int((page_h_pts - bbox.t) * scale)   # PDF top → pixel top
        px_bottom = int((page_h_pts - bbox.b) * scale)   # PDF bottom → pixel bottom

        # Safety clamp: pixel coordinates যেন image bounds এর মধ্যে থাকে
        img_w, img_h = img.size
        px_left   = max(0, min(px_left,   img_w))
        px_right  = max(0, min(px_right,  img_w))
        px_top    = max(0, min(px_top,    img_h))
        px_bottom = max(0, min(px_bottom, img_h))

        if px_right <= px_left or px_bottom <= px_top:
            print(f"    [WARN] Invalid crop box after conversion: "
                  f"({px_left},{px_top},{px_right},{px_bottom})")
            return False

        cropped = img.crop((px_left, px_top, px_right, px_bottom))
        cropped.save(output_path, format="PNG")
        return True

    except Exception as e:
        print(f"    [WARN] Crop failed: {e}")
        return False



# ─────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────
def extract_figures():
    # ── Setup ──
    print("=" * 60)
    print("Physics Book — Figure Extractor")
    print("=" * 60)

    if not PDF_PATH.exists():
        print(f"[ERROR] PDF not found: {PDF_PATH}")
        sys.exit(1)

    print(f"\n[1/4] Loading page map...")
    page_map = load_page_map()
    print(f"      {len(page_map)} pages mapped")

    print(f"\n[2/4] Setting up Docling (layout-only, no OCR)...")
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import DocItemLabel

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False  # table structure দরকার নেই এখন

    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=opts)}
    )
    print("      Docling ready")

    print(f"\n[3/4] Loading PDF with pypdfium2 (for cropping)...")
    import pypdfium2 as pdfium
    pdf_doc = pdfium.PdfDocument(str(PDF_PATH))
    total_pages = len(pdf_doc)
    print(f"      {total_pages} pages loaded")

    # ── Progress / Resume ──
    progress    = load_progress()
    done_pages  = set(progress["done_pages"])
    all_figures = progress["figures_found"]

    if done_pages:
        print(f"\n[RESUME] {len(done_pages)} pages already processed — continuing...")

    # ── Page-by-page processing ──
    print(f"\n[4/4] Processing pages...")
    print(f"      TORCHDYNAMO_DISABLE = {os.environ.get('TORCHDYNAMO_DISABLE', 'NOT SET')}")
    print()

    # একসাথে ৫ page করে batch এ process করব (memory efficient)
    BATCH_SIZE = 5

    for batch_start in range(0, total_pages, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_pages)
        batch_pages = list(range(batch_start, batch_end))  # 0-based

        # এই batch এ সব pages already done?
        if all(p in done_pages for p in batch_pages):
            pct = batch_end / total_pages * 100
            print(f"  [{pct:5.1f}%] Pages {batch_start+1}-{batch_end}: SKIP (already done)")
            continue

        pct = batch_end / total_pages * 100
        print(f"  [{pct:5.1f}%] Pages {batch_start+1}-{batch_end}...", end=" ", flush=True)

        try:
            # Sub-PDF বানাব এই batch এর জন্য
            sub_pdf = pdfium.PdfDocument.new()
            for page_idx in batch_pages:
                sub_pdf.import_pages(pdf_doc, pages=[page_idx])

            sub_bytes = io.BytesIO()
            sub_pdf.save(sub_bytes)
            sub_bytes.seek(0)

            # Docling দিয়ে layout analyze
            from docling.datamodel.document import DocumentStream
            ds = DocumentStream(name=f"batch_{batch_start}.pdf", stream=sub_bytes)
            result = converter.convert(ds)
            doc = result.document

            # Figure items খুঁজব
            batch_figures_count = 0
            for item, _ in doc.iterate_items():
                if item.label != DocItemLabel.PICTURE:
                    continue
                if not (hasattr(item, 'prov') and item.prov):
                    continue

                prov = item.prov[0]
                sub_page_idx = prov.page_no - 1  # Docling 1-based → 0-based
                if sub_page_idx >= len(batch_pages):
                    continue

                real_pdf_page_idx = batch_pages[sub_page_idx]   # 0-based actual page
                real_pdf_page_no  = real_pdf_page_idx + 1        # 1-based for display
                book_page         = page_map.get(real_pdf_page_no, -1)

                bbox = prov.bbox

                # Tiny figure reject করব
                area = bbox_area(bbox)
                if area < MIN_FIGURE_AREA:
                    continue

                # File নাম তৈরি
                fig_idx    = sum(1 for f in all_figures if f["pdf_page"] == real_pdf_page_no)
                image_name = f"fig_page{real_pdf_page_no:03d}_{fig_idx}.png"
                image_path = FIGURES_DIR / image_name

                # Crop করে save করব
                success = crop_and_save_figure(
                    pdf_doc, real_pdf_page_idx, bbox, image_path
                )

                if success:
                    all_figures.append({
                        "pdf_page":       real_pdf_page_no,
                        "book_page":      book_page,
                        "index_on_page":  fig_idx,
                        "bbox": {
                            "l": round(bbox.l, 2),
                            "t": round(bbox.t, 2),
                            "r": round(bbox.r, 2),
                            "b": round(bbox.b, 2),
                        },
                        "image_path":  str(image_path).replace("\\", "/"),
                        "figure_id":   "",   # 2_chunk.py এ fill হবে
                        "caption":     "",   # 2_chunk.py এ fill হবে
                    })
                    batch_figures_count += 1

            # Progress update
            for p in batch_pages:
                done_pages.add(p)
            progress["done_pages"]    = list(done_pages)
            progress["figures_found"] = all_figures
            save_progress(progress)

            print(f"OK ({batch_figures_count} figures)")

        except Exception as e:
            print(f"ERROR: {e}")
            # Continue with next batch — don't stop everything

    # ── Final save ──
    pdf_doc.close()

    # figure_locations.json এ সব save করব
    with open(LOCATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_figures, f, ensure_ascii=False, indent=2)

    # Progress file cleanup
    PROGRESS_FILE.unlink(missing_ok=True)

    print(f"\n{'=' * 60}")
    print(f"[DONE] Total figures extracted: {len(all_figures)}")
    print(f"[SAVED] {LOCATIONS_JSON}")
    print(f"[SAVED] Images in: {FIGURES_DIR}/")
    print(f"\nNext: python scripts/utils/content_detector.py")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    extract_figures()
