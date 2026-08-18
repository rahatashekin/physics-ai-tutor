"""
Test: Docling layout-only extraction on 3 pages
উদ্দেশ্য: figure bbox পাওয়া যায় কিনা দেখা
"""

import sys
import io
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

PDF_PATH = Path("data/raw/Secondary (BV)-2026_Class 9-10_Physics_compressed.pdf")

print("[1] Importing docling (may take 30s for model load)...")
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DocItemLabel
print("    OK")

print("[2] Setting pipeline options (no OCR)...")
opts = PdfPipelineOptions()
opts.do_ocr = False
opts.do_table_structure = False  # Table structure optional
print(f"    do_ocr = {opts.do_ocr}")

print("[3] Creating converter...")
converter = DocumentConverter(
    format_options={"pdf": PdfFormatOption(pipeline_options=opts)}
)
print("    OK")

print("[4] Converting pages 40-42 only (Chapter 2 region with figures)...")
# pypdfium2 diye specific pages extract korbo
import pypdfium2 as pdfium
import io as _io

pdf = pdfium.PdfDocument(str(PDF_PATH))
total = len(pdf)
print(f"    Total PDF pages: {total}")

# শুধু page 40-42 (index 39-41) নিয়ে একটা sub-PDF বানাব
sub_pdf = pdfium.PdfDocument.new()
for i in range(39, 42):  # pages 40, 41, 42
    sub_pdf.import_pages(pdf, pages=[i])

sub_bytes = _io.BytesIO()
sub_pdf.save(sub_bytes)
sub_bytes.seek(0)

print("[5] Converting sub-PDF with Docling...")
from docling.datamodel.document import DocumentStream

ds = DocumentStream(name="test_pages.pdf", stream=sub_bytes)
result = converter.convert(ds)
doc = result.document
print("    Done!")

print("\n[6] Analyzing elements found...")
label_counts = {}
figure_items = []

for item, level in doc.iterate_items():
    label = str(item.label)
    label_counts[label] = label_counts.get(label, 0) + 1

    if item.label == DocItemLabel.PICTURE:
        if hasattr(item, 'prov') and item.prov:
            prov = item.prov[0]
            figure_items.append({
                'page_no': prov.page_no,
                'bbox': prov.bbox if hasattr(prov, 'bbox') else None,
            })

print("\nLabel distribution:")
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count}")

print(f"\nFigures (PICTURE items) found: {len(figure_items)}")
for fig in figure_items:
    bbox = fig['bbox']
    if bbox:
        print(f"  Page {fig['page_no']}: l={bbox.l:.1f} t={bbox.t:.1f} r={bbox.r:.1f} b={bbox.b:.1f}")
    else:
        print(f"  Page {fig['page_no']}: bbox=N/A")

if figure_items:
    print("\n SUCCESS: Docling can detect figures without OCR!")
else:
    print("\n WARNING: No figures detected. May need do_ocr=True or different settings.")
