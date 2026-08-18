"""
PDF এর প্রথম কয়েক page এর raw text dump করা।
Chapter heading format বোঝার জন্য।
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pdfplumber
from pathlib import Path

PDF_PATH = Path(r"C:\Users\Home\Documents\PythonProjects\physics-tutor-agent\data\raw\Secondary (BV)-2026_Class 9-10_Physics_compressed.pdf")

SHOW_PAGES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # front matter / TOC
              20, 21, 22, 23, 24, 25,           # early content
              33, 34, 35,                        # expected ch2 start
              60, 61, 62,                        # expected ch2 end / ch3 start
             ]

with pdfplumber.open(PDF_PATH) as pdf:
    for pg_num in SHOW_PAGES:
        page = pdf.pages[pg_num - 1]
        text = page.extract_text() or "(empty)"
        print(f"\n{'='*60}")
        print(f"PDF PAGE {pg_num}")
        print('='*60)
        # first 40 lines
        for line in text.split("\n")[:40]:
            line = line.strip()
            if line:
                print(f"  {line}")
