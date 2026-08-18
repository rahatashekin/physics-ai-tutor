"""
scripts/utils/book_page_mapper.py
===================================
PURPOSE:
    extracted_book.md এ দুই ধরনের page number আছে:
      1. <!-- Page 48 -->   ← PDF এর page number (1-based)
      2. ৪৩                 ← বইয়ের ছাপানো page number (Bengali numeral)

    এই file টা দুটো match করে একটা dict বানায়:
        {pdf_page_num: book_page_num}
        e.g. {48: 43, 49: 44, ...}

STRATEGY:
    - প্রতিটা <!-- Page N --> section এর পরের ১০ line এ
      standalone Bengali numeral খুঁজব
    - Validation: 1-400 range, বাড়তে থাকা, year (2026) নয়
    - না পাওয়া গেলে: আগের mapping থেকে +1 করে interpolate করব

USAGE:
    from scripts.utils.book_page_mapper import load_page_map, pdf_to_book

    page_map = load_page_map()
    book_page = pdf_to_book(page_map, pdf_page=48)  # → 43
"""

import re
import json
from pathlib import Path

MD_PATH = Path("data/processed/extracted_book.md")
CACHE_PATH = Path("data/processed/page_map_cache.json")

# Bengali digit → Arabic digit
BN_TO_AR = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def bengali_to_int(text: str) -> int | None:
    """Bengali numeral string → int। Invalid হলে None।"""
    converted = text.strip().translate(BN_TO_AR)
    if converted.isdigit():
        return int(converted)
    return None


def is_valid_book_page(num: int, last_seen: int, pdf_page: int = 9999) -> bool:
    """
    Valid book page number কিনা check করে।
    - 1 থেকে 500 এর মধ্যে হতে হবে
    - আগের page এর চেয়ে বেশি হতে হবে (বা সর্বোচ্চ 2 কম)
    - 2026 এর মতো year হওয়া যাবে না
    - book_page > pdf_page + 5 হওয়া যাবে না (physical impossibility)
    """
    if not (1 <= num <= 500):
        return False
    if num == 2026:  # year print, very common in this book
        return False
    if last_seen > 0 and num < last_seen - 2:
        return False  # going backwards too much = probably not a page number
    if num > pdf_page + 5:
        return False  # impossible: book page can't be much greater than PDF page
    return True


def build_page_map(md_path: Path = MD_PATH) -> dict[int, int]:
    """
    extracted_book.md parse করে {pdf_page: book_page} dict বানায়।
    """
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    page_map: dict[int, int] = {}
    last_book_page = 0

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # <!-- Page N --> marker খুঁজছি
        m = re.match(r"<!--\s*Page\s+(\d+)\s*-->", line)
        if m:
            pdf_page = int(m.group(1))
            found_book_page = None

            # পরের ১০ line এ Bengali numeral খুঁজব
            for j in range(i + 1, min(i + 11, len(lines))):
                candidate = lines[j].strip()

                # Pure Bengali numeral (১, ২৮, ৩৪৫ etc.)
                # Reject: section numbers like "১.৫.১", years "২০২৬", mixed text
                if re.match(r"^[০-৯]+$", candidate):
                    num = bengali_to_int(candidate)
                    if num and is_valid_book_page(num, last_book_page, pdf_page):
                        found_book_page = num
                        last_book_page = num
                        break

                # আরবি numeral ও accept করব (some pages print Arabic)
                if re.match(r"^\d+$", candidate):
                    num = int(candidate)
                    if is_valid_book_page(num, last_book_page, pdf_page):
                        found_book_page = num
                        last_book_page = num
                        break

            if found_book_page:
                page_map[pdf_page] = found_book_page
            # না পাওয়া গেলে interpolation পরে করব

        i += 1

    # --- Monotonicity enforcement ---
    # যে pages monotonically increasing নয়, তাদের remove করব
    # তারপর interpolation তারা fill করবে
    page_map = _enforce_monotone(page_map)

    # --- Interpolation: missing pages fill করব ---
    page_map = _interpolate_missing(page_map)

    return page_map


def _enforce_monotone(page_map: dict[int, int]) -> dict[int, int]:
    """
    Page map monotonically increasing না হলে violating entries remove করে।
    Two-pass: forward pass removes drops, then check again.
    """
    if not page_map:
        return page_map

    # Forward pass
    sorted_items = sorted(page_map.items())
    clean: dict[int, int] = {}
    max_seen = 0

    for pdf_p, book_p in sorted_items:
        if book_p >= max_seen:
            clean[pdf_p] = book_p
            max_seen = book_p
        # else: skip this entry (violates monotonicity)

    return clean


def _interpolate_missing(page_map: dict[int, int]) -> dict[int, int]:
    """
    যেসব PDF pages এ book page number পাওয়া যায়নি,
    তাদের আগের এবং পরের known values থেকে interpolate করব।
    """
    if not page_map:
        return page_map

    all_pdf_pages = sorted(page_map.keys())
    min_pdf = all_pdf_pages[0]
    max_pdf = all_pdf_pages[-1]

    filled: dict[int, int] = dict(page_map)

    for pdf_p in range(min_pdf, max_pdf + 1):
        if pdf_p in filled:
            continue

        # আগের known page খুঁজব
        prev_pdf = max((p for p in filled if p < pdf_p), default=None)
        next_pdf = min((p for p in filled if p > pdf_p), default=None)

        if prev_pdf is not None and next_pdf is not None:
            # Linear interpolation
            prev_book = filled[prev_pdf]
            next_book = filled[next_pdf]
            ratio = (pdf_p - prev_pdf) / (next_pdf - prev_pdf)
            estimated = round(prev_book + ratio * (next_book - prev_book))
            filled[pdf_p] = estimated
        elif prev_pdf is not None:
            # শুধু আগের জানা আছে: +1 করে যাব
            filled[pdf_p] = filled[prev_pdf] + (pdf_p - prev_pdf)
        elif next_pdf is not None:
            filled[pdf_p] = filled[next_pdf] - (next_pdf - pdf_p)

    return filled


def load_page_map(use_cache: bool = True) -> dict[int, int]:
    """
    Page map load করে। Cache থাকলে cache থেকে, না থাকলে build করে।

    Args:
        use_cache: False দিলে force rebuild করবে

    Returns:
        {pdf_page: book_page} dict (keys and values are ints)
    """
    if use_cache and CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            raw = json.load(f)
            # JSON keys are always strings, convert back to int
            return {int(k): int(v) for k, v in raw.items()}

    print("[PageMap] Building page map from markdown...")
    page_map = build_page_map()

    # Cache save করব
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(page_map, f, indent=2)

    print(f"[PageMap] Done. Mapped {len(page_map)} PDF pages. Cache saved.")
    return page_map


def pdf_to_book(page_map: dict[int, int], pdf_page: int) -> int:
    """
    PDF page number → book page number।
    Mapping এ না থাকলে -1 return করে।
    """
    return page_map.get(pdf_page, -1)


# ---------------------------------------------------------------
# Standalone run: mapping verify করার জন্য
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os

    # Project root থেকে run করতে হবে
    os.chdir(Path(__file__).parent.parent.parent)

    print("Building page map (force rebuild)...")
    pm = load_page_map(use_cache=False)

    print(f"\nTotal mapped pages: {len(pm)}")
    print("\nSample mappings (first 20):")
    for pdf_p, book_p in sorted(pm.items())[:20]:
        print(f"  PDF page {pdf_p:3d}  ->  Book page {book_p}")

    print("\nSample mappings (middle):")
    items = sorted(pm.items())
    mid = len(items) // 2
    for pdf_p, book_p in items[mid:mid+10]:
        print(f"  PDF page {pdf_p:3d}  ->  Book page {book_p}")

    # Sanity check: monotonically increasing?
    values = [v for _, v in sorted(pm.items())]
    is_monotone = all(values[i] <= values[i+1] for i in range(len(values)-1))
    print(f"\nMonotonically increasing: {is_monotone}")
    if not is_monotone:
        print("WARNING: Some pages are out of order!")
        keys_sorted = sorted(pm.keys())
        for i in range(len(values)-1):
            if values[i] > values[i+1]:
                k1 = keys_sorted[i]
                k2 = keys_sorted[i+1]
                print(f"  PDF {k1}->{pm[k1]} then PDF {k2}->{pm[k2]}")
