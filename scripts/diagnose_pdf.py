import pypdfium2 as pdfium

doc = pdfium.PdfDocument(r"data/raw/Secondary (BV)-2026_Class 9-10_Physics_compressed.pdf")
print(f"Total pages: {len(doc)}")
print()

# Check multiple pages for embedded text
test_pages = [0, 5, 10, 30, 50, 100, 150, 200]
total_text = 0
for p in test_pages:
    page = doc[p]
    textpage = page.get_textpage()
    text = textpage.get_text_range().strip()
    total_text += len(text)
    status = "HAS TEXT" if text else "IMAGE ONLY"
    preview = repr(text[:60]) if text else "[empty]"
    print(f"Page {p+1:4d}: {status:12s} | {len(text):5d} chars | {preview}")

doc.close()
print()
print("VERDICT:", "TEXT-BASED PDF" if total_text > 100 else "SCANNED/IMAGE-BASED PDF — no embedded text")
