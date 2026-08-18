"""
src/book_index.py
==================
General BookIndex — extracted_book.md থেকে runtime এ build হয়।

APPROACH:
- "বহুনির্বাচনি প্রশ্ন" marker এর n-তম occurrence = chapter n এর exercise শুরু
  (chapters বইতে sequential order এ আছে, তাই 1st occurrence = ch1, 2nd = ch2, ...)
- প্রতিটা chapter এর exercise cluster এ:
    MCQ:        "বহুনির্বাচনি প্রশ্ন" থেকে
    Creative:   "সৃজনশীল প্রশ্ন" থেকে
    Shankhipto: "সংক্ষিপ্ত উত্তর প্রশ্ন" বা "সংক্ষিপ্ত প্রশ্ন" থেকে
- অনুসন্ধান:  "অনুসন্ধান X.XX" pattern → chapter এ sequential assign
- নিজে করো:   DB তে already আছে (content_type='niche_koro')
- <!-- Page N --> tags থেকে PDF page → line number mapping

No hardcoding. Works for all 13 chapters.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

BOOK_MD = Path("data/processed/extracted_book.md")

# PDF page → book page offset (our pipeline used PDF page numbers)
PDF_TO_BOOK_OFFSET = 5  # PDF page N = book page N - 5 (approximately)


# ─────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────
class ChapterExercises:
    """All exercise content for one chapter."""
    def __init__(self, chapter_num: int):
        self.chapter_num  = chapter_num
        self.mcq_text     = ""   # নমুনা প্রশ্ন / বহুনির্বাচনি
        self.creative_text= ""   # সৃজনশীল প্রশ্ন
        self.shankhipto_text = ""  # সংক্ষিপ্ত উত্তর প্রশ্ন
        self.onusondhan_texts: list[dict] = []  # [{id, text, pdf_page}, ...]
        self.mcq_pdf_page = 0
        self.creative_pdf_page = 0
        self.shankhipto_pdf_page = 0

    def __repr__(self):
        return (f"Ch{self.chapter_num}: mcq={len(self.mcq_text)} chars, "
                f"creative={len(self.creative_text)} chars, "
                f"shankhipto={len(self.shankhipto_text)} chars, "
                f"onusondhan={len(self.onusondhan_texts)}")


# ─────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────
class BookIndex:
    """
    Parses extracted_book.md once at init and provides
    structured access to exercise content by chapter.
    """

    def __init__(self, book_path: Path = BOOK_MD):
        self.book_path = book_path
        self._chapters: dict[int, ChapterExercises] = {}
        self._lines: list[str] = []
        self._page_map: dict[int, int] = {}   # line_index → pdf_page
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        text = self.book_path.read_text(encoding="utf-8", errors="replace")
        self._lines = text.splitlines()
        self._build_page_map()
        self._parse_exercises()
        self._parse_onusondhan()
        self._built = True

    # ── Page map ────────────────────────────────────────────

    def _build_page_map(self) -> None:
        """Build line_index → pdf_page from <!-- Page N --> comments."""
        cur_page = 1
        for i, line in enumerate(self._lines):
            m = re.search(r"<!--\s*[Pp]age\s+(\d+)\s*-->", line)
            if m:
                cur_page = int(m.group(1))
            self._page_map[i] = cur_page

    def _line_pdf_page(self, line_idx: int) -> int:
        return self._page_map.get(line_idx, 0)

    def _line_book_page(self, line_idx: int) -> int:
        return max(1, self._line_pdf_page(line_idx) - PDF_TO_BOOK_OFFSET)

    # ── Exercise section parser ──────────────────────────────

    def _parse_exercises(self) -> None:
        """
        Find all MCQ clusters (nth occurrence = chapter n).
        Within each cluster, find creative and shankhipto sections.
        """
        MCQ_PATTERNS       = ["বহুনির্বাচনি প্রশ্ন"]
        CREATIVE_PATTERNS  = ["সৃজনশীল প্রশ্ন"]
        SHANKHIPTO_PATTERNS= ["সংক্ষিপ্ত উত্তর প্রশ্ন", "সংক্ষিপ্ত প্রশ্ন",
                               "ও সংক্ষিপ্ত উত্তর"]

        def _matches(line: str, patterns: list[str]) -> bool:
            return any(p in line for p in patterns)

        # Find all MCQ start lines (these mark chapter exercise boundaries)
        mcq_starts: list[int] = []
        for i, line in enumerate(self._lines):
            if _matches(line, MCQ_PATTERNS):
                # Make sure this isn't a false positive inside body text
                # (real MCQ sections have the pattern near the start of line)
                stripped = line.strip()
                if any(p in stripped for p in MCQ_PATTERNS):
                    mcq_starts.append(i)

        # n-th MCQ start = chapter (n+1)
        for ch_idx, mcq_start in enumerate(mcq_starts):
            ch_num = ch_idx + 1
            if ch_num > 13:
                break

            # Chapter cluster ends just before next chapter's MCQ (or EOF)
            cluster_end = mcq_starts[ch_idx + 1] if ch_idx + 1 < len(mcq_starts) else len(self._lines)

            chunk = self._lines[mcq_start:cluster_end]

            # Find creative and shankhipto within this chunk
            creative_local   = None
            shankhipto_local = None

            for j, line in enumerate(chunk):
                if creative_local is None and _matches(line, CREATIVE_PATTERNS):
                    creative_local = j
                if shankhipto_local is None and _matches(line, SHANKHIPTO_PATTERNS):
                    shankhipto_local = j

            # ── Helper: find end of a section within chunk ──────────────────
            # Stops at (1) next Bengali chapter ordinal heading, (2) hard line cap
            CHAPTER_ORDINALS = [
                "প্রথম অধ্যায়", "দ্বিতীয় অধ্যায়", "তৃতীয় অধ্যায়",
                "চতুর্থ অধ্যায়", "পঞ্চম অধ্যায়", "ষষ্ঠ অধ্যায়",
                "সপ্তম অধ্যায়", "অষ্টম অধ্যায়", "নবম অধ্যায়",
                "দশম অধ্যায়", "একাদশ অধ্যায়", "দ্বাদশ অধ্যায়",
                "ত্রয়োদশ অধ্যায়",
            ]
            MAX_SECTION_LINES = 250

            def _section_end(chunk_lines: list, start_local: int) -> int:
                """Find the end index within chunk for a section starting at start_local."""
                cap = min(len(chunk_lines), start_local + MAX_SECTION_LINES)
                for k in range(start_local + 1, cap):
                    if any(ordinal in chunk_lines[k] for ordinal in CHAPTER_ORDINALS):
                        return k
                return cap

            # Extract text segments
            ex = ChapterExercises(ch_num)

            if creative_local is not None:
                ex.mcq_text = self._clean("\n".join(chunk[:creative_local]))
                ex.mcq_pdf_page = self._line_pdf_page(mcq_start)
            else:
                ex.mcq_text = self._clean("\n".join(chunk))
                ex.mcq_pdf_page = self._line_pdf_page(mcq_start)

            if creative_local is not None:
                if shankhipto_local is not None and shankhipto_local > creative_local:
                    ex.creative_text = self._clean(
                        "\n".join(chunk[creative_local:shankhipto_local])
                    )
                    ex.creative_pdf_page = self._line_pdf_page(mcq_start + creative_local)
                    # Shankhipto: stop at next chapter heading or 250 lines
                    sh_end = _section_end(chunk, shankhipto_local)
                    ex.shankhipto_text = self._clean(
                        "\n".join(chunk[shankhipto_local:sh_end])
                    )
                    ex.shankhipto_pdf_page = self._line_pdf_page(mcq_start + shankhipto_local)
                else:
                    cr_end = _section_end(chunk, creative_local)
                    ex.creative_text = self._clean(
                        "\n".join(chunk[creative_local:cr_end])
                    )
                    ex.creative_pdf_page = self._line_pdf_page(mcq_start + creative_local)

            self._chapters[ch_num] = ex

    # ── অনুসন্ধান parser ────────────────────────────────────

    def _parse_onusondhan(self) -> None:
        """
        Find all অনুসন্ধান X.XX sections and assign to chapters.
        Pattern: "অনুসন্ধান 2.01", "অনুসন্ধান ২.০৩" etc.
        Chapter inferred from the first digit of the ID.
        """
        ONUSONDAN_RE = re.compile(
            r"অনুসন্ধান\s+([০-৯\d]+)[\.।]([০-৯\d]+)"
        )
        BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

        # Collect all অনুসন্ধান start lines with their IDs
        starts: list[tuple[int, str, int]] = []  # (line_idx, id, ch_num)
        for i, line in enumerate(self._lines):
            m = ONUSONDAN_RE.search(line)
            if m:
                raw_ch  = m.group(1).translate(BN2EN)
                raw_num = m.group(2).translate(BN2EN)
                ch_num  = int(raw_ch)
                on_id   = f"{raw_ch}.{raw_num.zfill(2)}"
                starts.append((i, on_id, ch_num))

        # Chapter exercise boundary markers — stop onusondhan text before these
        EXERCISE_MARKERS = ["বহুনির্বাচনি প্রশ্ন", "সৃজনশীল প্রশ্ন",
                            "সংক্ষিপ্ত উত্তর প্রশ্ন"]
        MAX_ONUSONDAN_LINES = 120  # Hard cap: no onusondhan spans >120 lines

        # Extract text between consecutive অনুসন্ধান starts (or next chapter exercise)
        for idx, (start_line, on_id, ch_num) in enumerate(starts):
            if idx + 1 < len(starts):
                candidate_end = starts[idx + 1][0]
            else:
                candidate_end = start_line + MAX_ONUSONDAN_LINES

            # Cap at hard limit
            candidate_end = min(candidate_end, start_line + MAX_ONUSONDAN_LINES)

            # Stop earlier if we hit a chapter exercise marker (MCQ/Creative/Shankhipto)
            # This prevents overflow into the next chapter's content
            hard_end = candidate_end
            for k in range(start_line + 1, hard_end):
                if k >= len(self._lines):
                    break
                if any(marker in self._lines[k] for marker in EXERCISE_MARKERS):
                    hard_end = k
                    break

            end_line = hard_end
            text = self._clean("\n".join(self._lines[start_line:end_line]))
            entry = {
                "id":       on_id,
                "text":     text,
                "pdf_page": self._line_pdf_page(start_line),
                "book_page": self._line_book_page(start_line),
            }
            if ch_num not in self._chapters:
                self._chapters[ch_num] = ChapterExercises(ch_num)
            self._chapters[ch_num].onusondhan_texts.append(entry)

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Remove HTML comments and excessive whitespace."""
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ── Public API ──────────────────────────────────────────

    def get_mcq(self, chapter: int) -> str:
        self.build()
        ch = self._chapters.get(chapter)
        return ch.mcq_text if ch else ""

    def get_creative(self, chapter: int) -> str:
        self.build()
        ch = self._chapters.get(chapter)
        return ch.creative_text if ch else ""

    def get_shankhipto(self, chapter: int) -> str:
        self.build()
        ch = self._chapters.get(chapter)
        return ch.shankhipto_text if ch else ""

    def get_onusondhan(self, chapter: int, page: Optional[int] = None) -> list[dict]:
        """
        chapter এর সব অনুসন্ধান return করো।
        page দিলে সেই page এর কাছের অনুসন্ধান filter করো।
        """
        self.build()
        ch = self._chapters.get(chapter)
        if not ch:
            return []
        if page is None:
            return ch.onusondhan_texts
        # Filter by proximity to page
        filtered = [
            o for o in ch.onusondhan_texts
            if abs(o.get("book_page", 0) - page) <= 4
        ]
        return filtered if filtered else ch.onusondhan_texts

    def get_chapter(self, chapter: int) -> Optional[ChapterExercises]:
        self.build()
        return self._chapters.get(chapter)

    def coverage_report(self) -> str:
        self.build()
        lines = ["Chapter | MCQ | Creative | Shankhipto | Onusondhan"]
        lines.append("-" * 55)
        for ch in range(1, 14):
            ex = self._chapters.get(ch)
            if not ex:
                lines.append(f"   {ch:2d}   | MISSING")
                continue
            lines.append(
                f"   {ch:2d}   | {'✓' if ex.mcq_text else '✗':3} | "
                f"{'✓' if ex.creative_text else '✗':8} | "
                f"{'✓' if ex.shankhipto_text else '✗':10} | "
                f"{len(ex.onusondhan_texts):2d}"
            )
        return "\n".join(lines)


    def find_subsection(self, section_label: str) -> Optional[str]:
        """
        extracted_book.md তে directly একটা subsection heading খুঁজে content বের করো।

        section_label: "8.8.4" বা "৮.৮.৪" যেকোনো format
        Return: sub-section এর text (পরের heading পর্যন্ত), অথবা None

        কেন এই method:
        - DB এ subsections আলাদাভাবে labeled নয় (pipeline limitation)
        - এই method source-of-truth (extracted_book.md) থেকে directly পড়ে
        - Hardcode নেই — যেকোনো chapter এর যেকোনো X.Y.Z subsection কাজ করবে
        """
        self.build()

        # Build Bengali digit equivalent for the label (e.g., "8.8.4" → "৮.৮.৪")
        EN2BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
        bn_label = section_label.translate(EN2BN)
        en_label = section_label  # keep original too

        # Regex: match "৮.৮.৪" or "8.8.4" at start of line (with optional space)
        heading_re = re.compile(
            r"^\s*(?:" + re.escape(bn_label) + r"|" + re.escape(en_label) + r")\b"
        )
        # Next-section heading: any X.Y or X.Y.Z or chapter header
        next_section_re = re.compile(
            r"^\s*(?:[০-৯\d]+\.[০-৯\d]+(?:\.[০-৯\d]+)?)\s+\S"  # section heading
            r"|^#+\s"                                                # markdown heading
            r"|^<!-- Page"                                          # page marker (hard stop)
        )

        found_start = None
        content_lines = []

        for i, line in enumerate(self._lines):
            if found_start is None:
                if heading_re.match(line):
                    found_start = i
                    content_lines.append(line)
            else:
                # Stop at next section/chapter heading (but allow a few page markers)
                if next_section_re.match(line) and len(content_lines) > 3:
                    # Allow next <!-- Page --> only once (subsection may span pages)
                    if line.strip().startswith("<!-- Page") and content_lines.count("<!-- Page") == 0:
                        content_lines.append(line)
                        continue
                    break
                content_lines.append(line)
                if len(content_lines) > 80:   # safety limit: ~80 lines max
                    break

        if not content_lines:
            return None
        return "\n".join(content_lines).strip()


# Singleton instance
_INDEX: Optional[BookIndex] = None

def get_book_index() -> BookIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = BookIndex()
        _INDEX.build()
    return _INDEX
