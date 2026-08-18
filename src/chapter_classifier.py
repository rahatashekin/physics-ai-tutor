"""
src/chapter_classifier.py
===========================
Lightweight chapter inference from query keywords.

Approach (data-driven, no hardcoding):
  - Parse extracted_book.md to extract section headings per chapter
    e.g. "৮.৩ আলোর প্রতিফলন" → chapter 8
  - When a GENERAL query comes in without chapter context,
    score each chapter by how many query keywords appear in its section titles
  - If one chapter clearly dominates → use as filter in vector search
  - If ambiguous → don't filter (fall back to full-corpus search)

This works for ALL chapters because it's derived from the book's own content.
"""
import re
import pathlib
from typing import Optional
from functools import lru_cache

# Bengali digit → ASCII
_BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Pattern: Bengali section number at start of line e.g. "১.৩ ..." or "৮.১০ ..."
_SECTION_RE = re.compile(
    r"^([০-৯]+)\.([০-৯]+)\s+(.+)"
)

# Also capture English transliterations in parentheses
_PAREN_RE = re.compile(r"\(([^)]+)\)")

_BOOK_PATH = pathlib.Path("data/processed/extracted_book.md")


def _build_chapter_heading_index() -> dict[int, list[str]]:
    """
    Parse the book markdown and build:
      {chapter_num: [list of all section heading texts]}

    Each heading text is lowercased and includes both Bengali text
    and any English transliteration in parentheses.
    """
    index: dict[int, list[str]] = {}

    if not _BOOK_PATH.exists():
        return index

    with open(_BOOK_PATH, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        m = _SECTION_RE.match(line)
        if not m:
            continue
        ch_bn  = m.group(1)
        sec_bn = m.group(2)
        title  = m.group(3).strip()

        try:
            ch_num = int(ch_bn.translate(_BN2EN))
        except ValueError:
            continue

        if not (1 <= ch_num <= 13):
            continue

        # Collect both Bengali text and English words from parentheses
        combined = title
        for en in _PAREN_RE.findall(title):
            combined += " " + en

        # Remove parenthetical groups from main title (keep full for keyword matching)
        if ch_num not in index:
            index[ch_num] = []
        index[ch_num].append(combined)

    return index


@lru_cache(maxsize=1)
def _get_index() -> dict[int, list[str]]:
    """Cached chapter heading index — built once."""
    return _build_chapter_heading_index()


def infer_chapter(keywords: list[str], threshold: float = 0.35) -> Optional[int]:
    """
    Given a list of query keywords, infer which chapter they belong to.

    Returns:
      int  — chapter number if one chapter clearly dominates
      None — if ambiguous or no match (caller should search all chapters)

    Algorithm:
      For each chapter, count unique keywords that appear in ANY of its
      section headings. Normalize by total keywords.
      A chapter wins if its score >= threshold AND score > 2nd_best * 1.5
    """
    if not keywords:
        return None

    idx = _get_index()
    if not idx:
        return None

    scores: dict[int, float] = {}

    for ch, headings in idx.items():
        # Concatenate all headings into one searchable blob
        blob = " ".join(headings)
        matched = sum(1 for kw in keywords if kw in blob)
        if matched > 0:
            scores[ch] = matched / len(keywords)

    if not scores:
        return None

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_ch, best_score = ranked[0]

    if best_score < threshold:
        return None  # No chapter has enough keyword overlap

    # Check if winner is clearly ahead of runner-up
    if len(ranked) >= 2:
        _, second_score = ranked[1]
        if best_score < second_score * 1.5:
            return None  # Too close — ambiguous

    return best_ch


def infer_chapter_multi(
    keywords: list[str],
    threshold: float = 0.25,
    max_chapters: int = 2,
) -> list[int]:
    """
    Like infer_chapter but returns up to max_chapters candidates.
    Useful when the query spans topics from adjacent chapters.
    """
    if not keywords:
        return []

    idx = _get_index()
    if not idx:
        return []

    scores: dict[int, float] = {}
    for ch, headings in idx.items():
        blob = " ".join(headings)
        matched = sum(1 for kw in keywords if kw in blob)
        if matched > 0:
            scores[ch] = matched / len(keywords)

    candidates = [
        ch for ch, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if sc >= threshold
    ]
    return candidates[:max_chapters]


# ── Quick self-test (run directly) ────────────────────────────────────────────
if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    idx = _get_index()
    print(f"Chapter heading index built: {len(idx)} chapters")
    for ch in sorted(idx):
        print(f"  Ch={ch}: {len(idx[ch])} section headings")
        for h in idx[ch][:3]:
            print(f"    {h[:70]}")
    print()

    test_queries = [
        (["স্নেলের", "স্নেল", "সূত্র"],       9,  "স্নেলের সূত্র"),
        (["কুলম্বের", "কুলম্ব", "সূত্র"],      10, "কুলম্বের সূত্র"),
        (["ওহমের", "ওহম", "সূত্র"],             11, "ওহমের সূত্র"),
        (["ফ্যারাড", "আবেশ"],                   12, "ফ্যারাডের সূত্র"),
        (["ফটো ইলেকট্রিক", "আইনস্টাইন"],       13, "ফটো ইলেকট্রিক"),
        (["আর্কিমিডিস", "প্লব"],               5,  "আর্কিমিডিসের সূত্র"),
        (["বক্রতা", "আয়না", "দর্পণ"],         8,  "বক্রতার কেন্দ্র"),
        (["রৈখিক", "প্রসারণ"],                  6,  "রৈখিক প্রসারণ"),
        (["নিউটন", "ঘর্ষণ", "বল"],             3,  "নিউটনের সূত্র"),
        (["তাপ", "তাপমাত্রা"],                   6,  "তাপ ও তাপমাত্রা"),
        (["সূত্র"],                              None, "generic - ambiguous"),
        (["গতি", "বেগ", "ত্বরণ"],              2,  "গতির সমীকরণ"),
    ]

    print("Chapter inference test:")
    print("-" * 55)
    passed = 0
    for kws, expected, label in test_queries:
        got = infer_chapter(kws)
        ok  = "✅" if got == expected else "❌"
        print(f"  {ok} [{label}] → got={got} expected={expected}")
        if got == expected:
            passed += 1
    print(f"\n{passed}/{len(test_queries)} passed")
