"""
Quick sanity test: figure cropping on 5 pages (40-44)
"""
import io, sys, json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pypdfium2 as pdfium
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import DocumentStream
from docling_core.types.doc import DocItemLabel

from scripts.utils.figure_extractor import (
    PDF_PATH, FIGURES_DIR, MIN_FIGURE_AREA,
    load_page_map, bbox_area, crop_and_save_figure
)

print("Testing figure detection + cropping on PDF pages 40-44...")
print(f"PDF: {PDF_PATH}")
print()

opts = PdfPipelineOptions()
opts.do_ocr = False
opts.do_table_structure = False
conv = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opts)})

pdf_doc = pdfium.PdfDocument(str(PDF_PATH))
page_map = load_page_map()

# Pages 40-44 (0-based: 39-43) — Chapter 2 এর figure-heavy region
batch = list(range(39, 44))

print(f"Building sub-PDF for pages {[p+1 for p in batch]}...")
sub = pdfium.PdfDocument.new()
for i in batch:
    sub.import_pages(pdf_doc, pages=[i])
buf = io.BytesIO()
sub.save(buf)
buf.seek(0)

print("Running Docling layout analysis...")
ds = DocumentStream(name="test.pdf", stream=buf)
result = conv.convert(ds)
print("Done!\n")

found = []
all_items = list(result.document.iterate_items())
print(f"Total document items: {len(all_items)}")

for item, _ in all_items:
    if item.label != DocItemLabel.PICTURE:
        continue
    if not (hasattr(item, "prov") and item.prov):
        continue

    prov = item.prov[0]
    sub_idx = prov.page_no - 1  # Docling is 1-based
    if sub_idx >= len(batch):
        continue

    real_pdf_page = batch[sub_idx] + 1   # 1-based actual PDF page
    book_page     = page_map.get(real_pdf_page, -1)
    bbox          = prov.bbox
    area          = bbox_area(bbox)

    size_note = "SKIP (too small)" if area < MIN_FIGURE_AREA else "OK"
    print(f"  PDF page {real_pdf_page:3d} (book page {book_page:3d}) | "
          f"bbox=({bbox.l:.0f},{bbox.b:.0f},{bbox.r:.0f},{bbox.t:.0f}) | "
          f"area={area:.0f} | {size_note}")

    if area >= MIN_FIGURE_AREA:
        out = FIGURES_DIR / f"test_page{real_pdf_page:03d}.png"
        ok  = crop_and_save_figure(pdf_doc, batch[sub_idx], bbox, out)
        if ok:
            img_size = out.stat().st_size // 1024
            print(f"    -> Saved: {out.name} ({img_size} KB)")
            found.append(out)
        else:
            print(f"    -> FAILED to save")

pdf_doc.close()
print()
print(f"Total figures saved: {len(found)}")
if found:
    print("SUCCESS: Figure cropping is working!")
    print("\nSaved files:")
    for f in found:
        print(f"  {f}")
else:
    print("WARNING: No figures found/saved on these pages.")
    print("Try different page range if these pages don't have figures.")
