"""
scripts/utils/content_detector.py
===================================
PURPOSE:
    extracted_book.md এর text দেখে বলবে:
      - content_type: theory / example / niche_koro / onusondhan / mcq / etc.
      - chapter_num, section_num, sub_section_num
      - special IDs: onusondhan_id, figure_id, table_id, question_num, example_num

HOW IT WORKS:
    Stateful detection — একটা DetectorContext object আছে যেটা track করে
    আমরা এখন কোথায় আছি (কোন chapter, section, কোন mode এ)।
    Line by line process করার সময় context আপডেট হয়।

KEY DESIGN:
    detect_block(text, ctx) → (content_type, subtype, ids_dict, updated_ctx)

USAGE (2_chunk.py থেকে):
    from scripts.utils.content_detector import DetectorContext, detect_block, extract_ids
    
    ctx = DetectorContext()
    for block in markdown_blocks:
        content_type, subtype, ids, ctx = detect_block(block, ctx)
        # ids = {onusondhan_id, figure_id, table_id, question_num, ...}
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# Bengali ↔ Arabic numeral conversion
# ─────────────────────────────────────────────────────────────────
BN_TO_AR = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
AR_TO_BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

def bn_to_ar(text: str) -> str:
    """Bengali numeral string → Arabic numeral string।  '২.০৬' → '2.06'"""
    return text.translate(BN_TO_AR)

def to_int(text: str) -> Optional[int]:
    """Bengali বা Arabic numeral string → int।  Invalid হলে None।"""
    converted = bn_to_ar(text.strip())
    if converted.isdigit():
        return int(converted)
    return None

def section_to_arabic(section_str: str) -> str:
    """'২.৬.১' → '2.6.1'"""
    return bn_to_ar(section_str)


# ─────────────────────────────────────────────────────────────────
# Chapter ordinals (বাংলা → number)
# ─────────────────────────────────────────────────────────────────
CHAPTER_ORDINALS = {
    "প্রথম": 1, "দ্বিতীয়": 2, "তৃতীয়": 3, "চতুর্থ": 4,
    "পঞ্চম": 5, "ষষ্ঠ": 6, "সপ্তম": 7, "অষ্টম": 8,
    "নবম": 9, "দশম": 10, "একাদশ": 11, "দ্বাদশ": 12,
    "ত্রয়োদশ": 13, "চতুর্দশ": 14, "পঞ্চদশ": 15,
}

def parse_chapter_ordinal(text: str) -> Optional[int]:
    """'দ্বিতীয় অধ্যায়' → 2"""
    for bn_word, num in CHAPTER_ORDINALS.items():
        if bn_word in text:
            return num
    # Arabic numeral format: "অধ্যায় ২" বা "2nd chapter"
    m = re.search(r"অধ্যায়\s*([০-৯]+|\d+)", text)
    if m:
        return to_int(m.group(1))
    return None


# ─────────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────────

# Section number: "২.১", "২.৬.১", "2.1", "2.6.1"
# Bengali OR Arabic numerals, 2 or 3 levels deep
_NUM = r"[০-৯\d]+"
SECTION_RE = re.compile(
    rf"^({_NUM})\.({_NUM})(?:\.({_NUM}))?\s*(.*)$"
)

# Figure: "চিত্র ২.০৬" or "চিত্র ২.০৬:"
FIGURE_RE = re.compile(r"চিত্র\s*([০-৯\d]+[\.।][০-৯\d]+)")

# Table: "সারণি ২.০১" or "টেবিল ২.০১"
TABLE_RE = re.compile(r"(?:সারণি|টেবিল)\s*([০-৯\d]+[\.।][০-৯\d]+)")

# Onusondhan: "অনুসন্ধান ২.০১" or "অনুসন্ধান-২.০১"
ONUSONDHAN_RE = re.compile(r"অনুসন্ধান\s*[-।]?\s*([০-৯\d]+[\.।][০-৯\d]+)")

# Example number: "উদাহরণ ১" or "উদাহরণ-১"
EXAMPLE_NUM_RE = re.compile(r"উদাহরণ\s*[-।]?\s*([০-৯\d]+)")

# Question number at line start: "১।" or "১." or "১)" 
QUESTION_NUM_RE = re.compile(r"^([০-৯\d]+)[।\.\)]")

# Formula indicators: contains = sign and math-like content
FORMULA_RE = re.compile(r"[a-zA-Zα-ωΑ-Ω\u09a6-\u09f3]*\s*=\s*[^=]")


# ─────────────────────────────────────────────────────────────────
# DetectorContext — stateful tracker
# ─────────────────────────────────────────────────────────────────
@dataclass
class DetectorContext:
    """
    Processing করার সময় কোথায় আছি সেটা track করে।
    Line-by-line বা block-by-block আপডেট হয়।
    """
    # ── Chapter / Section ──
    chapter_num:       int  = 0
    chapter_title:     str  = ""
    section_num:       str  = ""    # "2.6"
    section_title:     str  = ""
    sub_section_num:   str  = ""    # "2.6.1"
    sub_section_title: str  = ""

    # ── Current mode (কোন section type এ আছি) ──
    in_example:         bool = False
    example_num:        int  = -1
    in_onusondhan:      bool = False
    onusondhan_id:      str  = ""
    in_niche_koro:      bool = False
    in_doliya_kaj:      bool = False

    # ── Exercise section mode ──
    in_mcq_section:     bool = False
    in_nomuna_block:    bool = False   # নমুনা প্রশ্ন block (MCQ এর ভেতরে)
    in_cq_section:      bool = False
    in_sq_section:      bool = False

    # ── Paragraph counter (section/page reset হয়) ──
    para_index:         int  = 0

    def reset_exercise_modes(self):
        """নতুন exercise section শুরু হলে আগেরটা বন্ধ করি"""
        self.in_mcq_section  = False
        self.in_nomuna_block = False
        self.in_cq_section   = False
        self.in_sq_section   = False

    def reset_content_modes(self):
        """নতুন section শুরু হলে content modes reset করি"""
        self.in_example      = False
        self.example_num     = -1
        self.in_onusondhan   = False
        self.onusondhan_id   = ""
        self.in_niche_koro   = False
        self.in_doliya_kaj   = False
        self.para_index      = 0


# ─────────────────────────────────────────────────────────────────
# Main detection function
# ─────────────────────────────────────────────────────────────────
def detect_block(text: str, ctx: DetectorContext) -> tuple[str, str, dict, DetectorContext]:
    """
    একটা text block এর content_type detect করে।

    Args:
        text: block এর text (could be multiple lines)
        ctx:  current context (আপডেট হবে)

    Returns:
        (content_type, content_subtype, ids_dict, updated_ctx)

        content_type: "theory" / "example" / "niche_koro" / "onusondhan" / 
                      "mcq" / "mcq_nomuna" / "creative_question" / 
                      "short_question" / "figure" / "table" / 
                      "learning_objectives" / "chapter_intro" /
                      "section_header" / "sub_section_header" /
                      "formula" / "biography" / "doliya_kaj"

        content_subtype: "question" / "answer" / "procedure" / "" etc.

        ids_dict: {
            "onusondhan_id": "2.01",
            "figure_id": "2.06",
            "table_id": "2.01",
            "question_num": 3,
            "example_num": 1,
            "is_nomuna": False,
        }
    """
    stripped = text.strip()
    first_line = stripped.split("\n")[0].strip()
    ids = _empty_ids()

    # ── 1. Chapter header ──
    if "অধ্যায়" in first_line:
        chapter_num = parse_chapter_ordinal(first_line)
        if chapter_num:
            ctx.chapter_num   = chapter_num
            ctx.chapter_title = _extract_chapter_title(stripped)
            ctx.section_num   = ""
            ctx.section_title = ""
            ctx.reset_content_modes()
            ctx.reset_exercise_modes()
            return "chapter_intro", "", ids, ctx

    # ── 2. শেখার উদ্দেশ্য ──
    if "শেষে আমরা" in stripped or "শেখার উদ্দেশ্য" in stripped:
        return "learning_objectives", "", ids, ctx

    # ── 3. জীবনী / biography ──
    if any(kw in stripped for kw in ["জীবনী", "বিজ্ঞানীর", "আবিষ্কারক"]):
        return "biography", "", ids, ctx

    # ── 4. Exercise sections (শেষের অধ্যায় exercises) ──
    if "বহুনির্বাচনি প্রশ্ন" in stripped:
        ctx.reset_exercise_modes()
        ctx.in_mcq_section = True
        return "mcq", "header", ids, ctx

    if "নমুনা প্রশ্ন" in stripped and ctx.in_mcq_section:
        ctx.in_nomuna_block = True
        return "mcq_nomuna", "header", ids, ctx

    if "সৃজনশীল প্রশ্ন" in stripped:
        ctx.reset_exercise_modes()
        ctx.in_cq_section = True
        return "creative_question", "header", ids, ctx

    if "সংক্ষিপ্ত উত্তর" in stripped:
        ctx.reset_exercise_modes()
        ctx.in_sq_section = True
        return "short_question", "header", ids, ctx

    # ── 5. Section header (X.Y or X.Y.Z) ──
    # NOTE: এটা question number check এর আগে করতে হবে
    # কারণ "২.১ গতি" এর মতো section এ QUESTION_NUM_RE false match করে
    sec_match = SECTION_RE.match(first_line)
    if sec_match:
        lvl1, lvl2, lvl3, title = sec_match.groups()
        lvl1_ar = bn_to_ar(lvl1)
        lvl2_ar = bn_to_ar(lvl2)

        # ── SANITY CHECK ──
        # Section numbers must be reasonable for a textbook:
        # Chapter: 1-15, Section: 1-30, Sub-section: 1-10
        # এই range এর বাইরে হলে এটা page number বা অন্য কিছু, section নয়
        lvl1_int = int(lvl1_ar) if lvl1_ar.isdigit() else 999
        lvl2_int = int(lvl2_ar) if lvl2_ar.isdigit() else 999
        lvl3_int = int(bn_to_ar(lvl3)) if lvl3 and bn_to_ar(lvl3).isdigit() else 0

        is_valid_section = (
            1 <= lvl1_int <= 15 and
            1 <= lvl2_int <= 30 and
            lvl3_int <= 10
        )

        if not is_valid_section:
            # NOT a section header — fall through to other detections
            pass
        elif lvl3:
            # Sub-section: 2.6.1
            lvl3_ar = bn_to_ar(lvl3)
            ctx.sub_section_num   = f"{lvl1_ar}.{lvl2_ar}.{lvl3_ar}"
            ctx.sub_section_title = title.strip()
            ctx.reset_content_modes()
            return "sub_section_header", "", ids, ctx
        else:
            # Section: 2.6
            new_section = f"{lvl1_ar}.{lvl2_ar}"
            if new_section != ctx.section_num:
                ctx.reset_content_modes()
            ctx.section_num      = new_section
            ctx.section_title    = title.strip()
            ctx.sub_section_num  = ""
            ctx.sub_section_title = ""
            return "section_header", "", ids, ctx


    # ── 6. MCQ / CQ / SQ individual question ──
    q_match = QUESTION_NUM_RE.match(first_line)
    if q_match:
        q_num = to_int(q_match.group(1))
        ids["question_num"] = q_num

        if ctx.in_nomuna_block:
            ids["is_nomuna"] = True
            return "mcq_nomuna", "question", ids, ctx
        if ctx.in_mcq_section:
            return "mcq", "question", ids, ctx
        if ctx.in_cq_section:
            return "creative_question", "question", ids, ctx
        if ctx.in_sq_section:
            return "short_question", "question", ids, ctx

    # ── 6. অনুসন্ধান ──
    # first_word check: "অনুসন্ধান" line এর শুরুতে থাকলেই detect করব
    # যাতে "এই অনুসন্ধান করো" টাইপের theory line false detect না করে
    _first_word = first_line.strip().split()[0] if first_line.strip() else ""
    on_match = ONUSONDHAN_RE.search(stripped)
    if on_match or _first_word == "অনুসন্ধান":
        ctx.in_onusondhan = True
        ctx.in_example    = False
        ctx.in_niche_koro = False
        if on_match:
            raw_id = bn_to_ar(on_match.group(1))
            ctx.onusondhan_id   = raw_id
            ids["onusondhan_id"] = raw_id
        return "onusondhan", "", ids, ctx

    # ── 7. উদাহরণ ──
    # first_word check: "উদাহরণ ১" বা "উদাহরণ-২" এর মতো line এর শুরুতে থাকলেই example
    # Guard: অনুসন্ধান এর ভেতরে থাকলে "উদাহরণ" investigation এর অংশ, escape নয়
    if _first_word == "উদাহরণ" and not ctx.in_onusondhan:
        ctx.in_example    = True
        ctx.in_onusondhan = False
        ctx.in_niche_koro = False
        ex_match = EXAMPLE_NUM_RE.search(first_line)
        if ex_match:
            ex_num = to_int(ex_match.group(1))
            ctx.example_num    = ex_num if ex_num else -1
            ids["example_num"] = ctx.example_num
        return "example", "header", ids, ctx

    # ── 8. নিজে করো — ESCAPE keyword, must be before in_example check ──
    # Guard: in_onusondhan mode এ থাকলে "নিজে করো" investigation এর step instruction,
    # নতুন activity নয় — escape করব না
    if "নিজে করো" in stripped and not ctx.in_onusondhan:
        ctx.in_niche_koro = True
        ctx.in_example    = False
        ctx.in_onusondhan = False
        ctx.in_doliya_kaj = False
        return "niche_koro", "", ids, ctx

    # ── 9. দলীয় কাজ — ESCAPE keyword ──
    # Guard: in_onusondhan mode এ থাকলে escape করব না
    if "দলীয় কাজ" in stripped and not ctx.in_onusondhan:
        ctx.in_doliya_kaj = True
        ctx.in_niche_koro = False
        ctx.in_example    = False
        ctx.in_onusondhan = False
        return "doliya_kaj", "", ids, ctx

    # Within example: detect sub-parts
    if ctx.in_example:
        if "প্রশ্ন" in first_line or first_line.startswith("প্র"):
            ids["example_num"] = ctx.example_num
            return "example", "question", ids, ctx
        if "সমাধান" in first_line or "উত্তর" in first_line:
            ids["example_num"] = ctx.example_num
            return "example", "answer", ids, ctx
        # Still inside example (continuation)
        ids["example_num"] = ctx.example_num
        return "example", "body", ids, ctx

    # ── 11. Figure ──
    fig_match = FIGURE_RE.search(stripped)
    if fig_match:
        raw_id = bn_to_ar(fig_match.group(1))
        ids["figure_id"] = raw_id
        return "figure", "", ids, ctx

    # ── 12. Table ──
    tbl_match = TABLE_RE.search(stripped)
    if tbl_match:
        raw_id = bn_to_ar(tbl_match.group(1))
        ids["table_id"] = raw_id
        return "table", "", ids, ctx

    # ── 13. Formula (heuristic) ──
    if FORMULA_RE.search(stripped) and len(stripped) < 200:
        # Short line with equation pattern → formula
        return "formula", "", ids, ctx

    # ── 14. Currently inside a special mode ──
    if ctx.in_onusondhan:
        ids["onusondhan_id"] = ctx.onusondhan_id
        return "onusondhan", "body", ids, ctx

    if ctx.in_niche_koro:
        return "niche_koro", "body", ids, ctx

    if ctx.in_doliya_kaj:
        return "doliya_kaj", "body", ids, ctx

    # ── 15. Default: theory ──
    ctx.para_index += 1
    return "theory", "", ids, ctx


# ─────────────────────────────────────────────────────────────────
# Helper: extract all IDs from a text block
# ─────────────────────────────────────────────────────────────────
def extract_ids(text: str) -> dict:
    """
    Text থেকে সব special IDs একসাথে extract করে।
    detect_block এর পাশাপাশি standalone use করা যায়।
    """
    ids = _empty_ids()

    fig = FIGURE_RE.search(text)
    if fig:
        ids["figure_id"] = bn_to_ar(fig.group(1))

    tbl = TABLE_RE.search(text)
    if tbl:
        ids["table_id"] = bn_to_ar(tbl.group(1))

    on = ONUSONDHAN_RE.search(text)
    if on:
        ids["onusondhan_id"] = bn_to_ar(on.group(1))

    ex = EXAMPLE_NUM_RE.search(text)
    if ex:
        ids["example_num"] = to_int(ex.group(1)) or -1

    q = QUESTION_NUM_RE.match(text.strip())
    if q:
        ids["question_num"] = to_int(q.group(1)) or -1

    return ids


def _empty_ids() -> dict:
    return {
        "onusondhan_id": "",
        "figure_id":     "",
        "table_id":      "",
        "question_num":  -1,
        "example_num":   -1,
        "is_nomuna":     False,
    }


def _extract_chapter_title(text: str) -> str:
    """Chapter header text থেকে title extract করে।"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # অধ্যায় line এর পরের line টা usually chapter title
    for i, line in enumerate(lines):
        if "অধ্যায়" in line and i + 1 < len(lines):
            return lines[i + 1]
    return ""


# ─────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, io, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    # Windows terminal UTF-8 fix
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    # Test cases — বইয়ের actual patterns
    test_cases = [
        ("দ্বিতীয় অধ্যায়\nগতি", "→ chapter_intro"),
        ("এ অধ্যায় শেষে আমরা যা শিখব", "→ learning_objectives"),
        ("২.১ গতি কাকে বলে?", "→ section_header (2.1)"),
        ("২.৬ ত্বরণ (Acceleration)", "→ section_header (2.6)"),
        ("২.৬.১ সুষম ত্বরণ", "→ sub_section_header (2.6.1)"),
        ("উদাহরণ ১\nপ্রশ্ন: একটি গাড়ি...", "→ example"),
        ("নিজে করো\nএকটি বল ছুঁড়ে দাও...", "→ niche_koro"),
        ("অনুসন্ধান ২.০১\nউপকরণ: একটি...", "→ onusondhan (id=2.01)"),
        ("চিত্র ২.০৬: বেগ-সময় লেখচিত্র", "→ figure (id=2.06)"),
        ("সারণি ২.০১: বিভিন্ন বস্তুর ত্বরণ", "→ table (id=2.01)"),
        ("বহুনির্বাচনি প্রশ্ন", "→ mcq header"),
        ("নমুনা প্রশ্ন", "→ mcq_nomuna header"),
        ("১। নিচের কোনটি সঠিক?", "→ mcq question (num=1)"),
        ("সৃজনশীল প্রশ্ন", "→ creative_question header"),
        ("৩। রহিম সাহেব একটি গাড়িতে...", "→ creative_question (num=3)"),
        ("v = u + at", "→ formula"),
        # theory: প্রথমে section রেসেট দরকার (নর্মাল book flow এ section change হলে context reset হয়)
        ("২.৯ গতির প্রকার", "→ section_header (resets modes)"),
        ("বেগ হলো একটি ভেক্টর রাশি যা...", "→ theory"),
    ]

    print("=" * 60)
    print("Content Detector — Test Results")
    print("=" * 60)

    # Sequential test (context carries over)
    ctx = DetectorContext()
    for text, expected in test_cases:
        ctype, subtype, ids, ctx = detect_block(text, ctx)
        subtype_str = f"/{subtype}" if subtype else ""
        ids_str = ""
        for k, v in ids.items():
            if v and v != -1 and v is not False:
                ids_str += f" {k}={v}"

        print(f"\n  Input:    {text[:50]!r}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {ctype}{subtype_str}{ids_str}")
        print(f"  Context:  ch={ctx.chapter_num} sec={ctx.section_num} "
              f"mcq={ctx.in_mcq_section} ex={ctx.in_example}")

    print("\n" + "=" * 60)
    print("Chapter context at end:")
    print(f"  chapter_num: {ctx.chapter_num}")
    print(f"  section_num: {ctx.section_num}")
    print(f"  in_mcq_section: {ctx.in_mcq_section}")
