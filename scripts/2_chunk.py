"""
scripts/2_chunk.py
==================
PURPOSE:
    extracted_book.md → parent_chunks.jsonl + child_chunks.jsonl

STRATEGY (3-way):
    ভাগ ১ — Complete Units (অনুসন্ধান, নিজে করো, Table, Figure)
        → শুধু parent_chunks এ, child নেই
        
    ভাগ ২ — Exact Lookup (MCQ, CQ, SQ, নমুনা)
        → প্রতিটা question = 1 parent chunk, metadata filter দিয়ে খুঁজবে
        
    ভাগ ৩ — Semantic (Theory, Formula)
        → child (paragraph) + parent (section-level accumulation)

PIPELINE:
    1. Parse markdown → Segments (content-type-homogeneous pieces)
    2. Match figures (চিত্র ২.০৬ → PNG path)
    3. Build parent + child chunks
    4. Save as JSONL

USAGE:
    python scripts/2_chunk.py
"""

import io
import json
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# Project root
# ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.content_detector import (
    DetectorContext, detect_block,
    FIGURE_RE, TABLE_RE, ONUSONDHAN_RE,
    bn_to_ar,
)
from scripts.utils.book_page_mapper import load_page_map

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MARKDOWN_FILE    = Path("data/processed/extracted_book.md")
FIGURE_JSON      = Path("data/processed/figure_locations.json")
OUTPUT_DIR       = Path("data/processed")
PARENT_CHUNKS    = OUTPUT_DIR / "parent_chunks.jsonl"
CHILD_CHUNKS     = OUTPUT_DIR / "child_chunks.jsonl"

# Theory paragraph তে token কত হলে পরের child chunk শুরু হবে
# rough estimate: 1 token ≈ 3 chars (Bengali)
MAX_CHILD_CHARS  = 600   # ≈ 200 tokens

# Complete Unit types — এগুলো ভাগ করা যাবে না
COMPLETE_UNIT_TYPES = {
    "onusondhan", "niche_koro", "doliya_kaj",
    "learning_objectives", "biography",
}

# Exercise types — Exact Lookup (প্রতিটা question = 1 parent)
EXERCISE_TYPES = {
    "mcq", "mcq_nomuna", "creative_question", "short_question",
}

# Semantic types — parent + child দুটোই লাগবে
SEMANTIC_TYPES = {
    "theory", "formula", "example",
}


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
@dataclass
class Segment:
    """একটা content-type-homogeneous piece। Segments থেকে Chunks তৈরি হবে।"""
    content_type:     str
    content_subtype:  str
    text:             str

    chapter_num:      int = 0
    chapter_title:    str = ""
    section_num:      str = ""
    section_title:    str = ""
    sub_section_num:  str = ""

    pdf_page_start:   int = -1
    pdf_page_end:     int = -1
    book_page_start:  int = -1
    book_page_end:    int = -1

    onusondhan_id:    str  = ""
    figure_id:        str  = ""
    figure_image_path:str  = ""
    table_id:         str  = ""
    question_num:     int  = -1
    example_num:      int  = -1
    is_nomuna:        bool = False


@dataclass
class Chunk:
    """Final chunk — LanceDB তে store হবে।"""
    chunk_id:          str
    parent_id:         str
    chunk_type:        str   # "parent" / "child"

    content_type:      str
    content_subtype:   str
    text:              str

    chapter_num:       int
    chapter_title:     str
    section_num:       str
    section_title:     str
    sub_section_num:   str

    book_page_start:   int
    book_page_end:     int
    spans_pages:       bool

    onusondhan_id:     str
    figure_id:         str
    figure_image_path: str
    table_id:          str
    question_num:      int
    example_num:       int
    is_nomuna:         bool
    token_count:       int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────
# Phase 1: Parse markdown → Segments
# ─────────────────────────────────────────────
PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s*(\d+)\s*-->")


def parse_markdown_to_segments(md_path: Path, page_map: dict[int, int]) -> list[Segment]:
    """extracted_book.md পড়ে Segment list তৈরি করে।"""
    print("[Phase 1] Parsing markdown to segments...")

    segments: list[Segment] = []
    ctx = DetectorContext()

    current_pdf_page  = 0
    current_book_page = 0

    block_lines:      list[str] = []
    block_pdf_start:  int = 0
    block_book_start: int = 0
    block_type:       str = ""
    block_subtype:    str = ""
    block_ids:        dict = {}

    def flush_block():
        nonlocal block_lines, block_pdf_start, block_book_start
        nonlocal block_type, block_subtype, block_ids

        text = "\n".join(block_lines).strip()
        if not text or not block_type:
            block_lines = []
            return

        seg = Segment(
            content_type    = block_type,
            content_subtype = block_subtype,
            text            = text,
            chapter_num     = ctx.chapter_num,
            chapter_title   = ctx.chapter_title,
            section_num     = ctx.section_num,
            section_title   = ctx.section_title,
            sub_section_num = ctx.sub_section_num,
            pdf_page_start  = block_pdf_start,
            pdf_page_end    = current_pdf_page,
            book_page_start = block_book_start,
            book_page_end   = current_book_page,
            onusondhan_id   = block_ids.get("onusondhan_id", ""),
            figure_id       = block_ids.get("figure_id", ""),
            table_id        = block_ids.get("table_id", ""),
            question_num    = block_ids.get("question_num", -1),
            example_num     = block_ids.get("example_num", -1),
            is_nomuna       = block_ids.get("is_nomuna", False),
        )
        segments.append(seg)
        block_lines = []

    with open(md_path, encoding="utf-8") as f:
        all_lines = f.readlines()

    total = len(all_lines)
    print(f"  Total lines: {total}")

    for i, raw_line in enumerate(all_lines):
        line = raw_line.rstrip("\n")

        # Page marker
        pm = PAGE_MARKER_RE.match(line.strip())
        if pm:
            new_pdf_page  = int(pm.group(1))
            new_book_page = page_map.get(new_pdf_page, current_book_page)

            if block_type in COMPLETE_UNIT_TYPES:
                # Complete Unit: page boundary তে flush করব না!
                # অনুসন্ধান/নিজে করো multi-page হতে পারে — একটাই chunk থাকবে
                current_pdf_page  = new_pdf_page
                current_book_page = new_book_page
            else:
                # Theory/Semantic: page change এ flush করি
                if block_lines:
                    flush_block()
                current_pdf_page  = new_pdf_page
                current_book_page = new_book_page
                if block_pdf_start == 0:
                    block_pdf_start  = current_pdf_page
                    block_book_start = current_book_page
            continue


        # Blank line
        if not line.strip():
            if block_type in SEMANTIC_TYPES and block_lines:
                flush_block()
                block_pdf_start  = current_pdf_page
                block_book_start = current_book_page
            elif block_type in COMPLETE_UNIT_TYPES and block_lines:
                block_lines.append("")
            continue

        # Detect line type
        ctype, subtype, ids, ctx = detect_block(line, ctx)

        # ── Type change detection ──
        # Complete Unit (অনুসন্ধান, নিজে করো, দলীয় কাজ, etc.) এর ভেতরে
        # formula/theory/figure দেখলে split করব না।
        # শুধু "hard boundary" এ split করব:
        #   - নতুন chapter বা section শুরু হলে
        #   - ভিন্ন Complete Unit শুরু হলে (যেমন অনুসন্ধান এর পরে নিজে করো)
        #   - Exercise section header আসলে
        HARD_BOUNDARY_TYPES = {"chapter_intro", "section_header", "sub_section_header"}

        if block_type in COMPLETE_UNIT_TYPES:
            # Complete Unit mode: শুধু hard boundary তে flush
            type_changed = (
                ctype in HARD_BOUNDARY_TYPES
                or (ctype in EXERCISE_TYPES and subtype == "header")
                or (ctype in COMPLETE_UNIT_TYPES and ctype != block_type)
                # একই type কিন্তু নতুন instance:
                # (যেমন অনুসন্ধান ১.০১ এর পরে অনুসন্ধান ১.০২)
                or (
                    ctype == block_type
                    and ctype == "onusondhan"
                    and ids.get("onusondhan_id")
                    and block_ids.get("onusondhan_id")
                    and ids.get("onusondhan_id") != block_ids.get("onusondhan_id")
                )
            )
        else:
            # Regular mode: যেকোনো type change এ flush
            type_changed = (ctype != block_type) or (
                ctype in EXERCISE_TYPES
                and subtype == "question"
                and block_subtype == "question"
                and ids.get("question_num", -1) != block_ids.get("question_num", -1)
            )

        if type_changed and block_lines:
            flush_block()
            block_pdf_start  = current_pdf_page
            block_book_start = current_book_page


        if not block_lines:
            block_type    = ctype
            block_subtype = subtype
            block_ids     = dict(ids)
            if block_pdf_start == 0:
                block_pdf_start  = current_pdf_page
                block_book_start = current_book_page
        else:
            for k, v in ids.items():
                if v and v != -1 and v is not False and not block_ids.get(k):
                    block_ids[k] = v

        block_lines.append(line)

        if i % 2000 == 0:
            pct = i / total * 100
            print(f"  [{pct:5.1f}%] {i}/{total} lines, {len(segments)} segments")

    if block_lines:
        flush_block()

    print(f"  Done. {len(segments)} segments created.")
    return segments


# ─────────────────────────────────────────────
# Phase 2: Figure matching
# ─────────────────────────────────────────────
def build_figure_lookup(figure_json_path: Path) -> dict[int, list[dict]]:
    """figure_locations.json → book_page → [figures] lookup"""
    if not figure_json_path.exists():
        print("[WARN] figure_locations.json not found. Skipping figure matching.")
        return {}

    with open(figure_json_path, encoding="utf-8") as f:
        figures = json.load(f)

    lookup: dict[int, list[dict]] = {}
    for fig in figures:
        bp = fig.get("book_page", -1)
        if bp > 0:
            lookup.setdefault(bp, []).append(fig)

    print(f"[Phase 2] Figure lookup built: {len(figures)} figures, {len(lookup)} book pages")
    return lookup


def match_figure_to_segment(seg: Segment, fig_lookup: dict[int, list[dict]]) -> str:
    """Segment এর book page দেখে matching PNG path খোঁজে।"""
    for bp in [seg.book_page_start, seg.book_page_start + 1]:
        figs = fig_lookup.get(bp, [])
        if figs:
            return figs[0].get("image_path", "")
    return ""


# ─────────────────────────────────────────────
# Phase 3: Build chunks
# ─────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def make_chunk_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def split_into_paragraphs(text: str, max_chars: int = MAX_CHILD_CHARS) -> list[str]:
    """Long theory text → paragraph list"""
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs = []
    for para in raw_paras:
        if len(para) <= max_chars:
            paragraphs.append(para)
        else:
            lines = [l.strip() for l in para.split("\n") if l.strip()]
            current: list[str] = []
            current_len = 0
            for ln in lines:
                if current_len + len(ln) > max_chars and current:
                    paragraphs.append("\n".join(current))
                    current = [ln]
                    current_len = len(ln)
                else:
                    current.append(ln)
                    current_len += len(ln)
            if current:
                paragraphs.append("\n".join(current))
    return paragraphs or [text]


def _make_parent(seg: Segment, ctype: str = "", extra: dict = {}) -> Chunk:
    """Helper: Segment থেকে parent Chunk তৈরি।"""
    ct = ctype or seg.content_type
    prefix = f"ch{seg.chapter_num}_{ct}"
    pid = make_chunk_id(prefix)
    return Chunk(
        chunk_id          = pid,
        parent_id         = pid,
        chunk_type        = "parent",
        content_type      = ct,
        content_subtype   = seg.content_subtype,
        text              = seg.text,
        chapter_num       = seg.chapter_num,
        chapter_title     = seg.chapter_title,
        section_num       = seg.section_num,
        section_title     = seg.section_title,
        sub_section_num   = seg.sub_section_num,
        book_page_start   = seg.book_page_start,
        book_page_end     = seg.book_page_end,
        spans_pages       = (seg.book_page_start != seg.book_page_end),
        onusondhan_id     = extra.get("onusondhan_id", seg.onusondhan_id),
        figure_id         = extra.get("figure_id", seg.figure_id),
        figure_image_path = extra.get("figure_image_path", seg.figure_image_path),
        table_id          = extra.get("table_id", seg.table_id),
        question_num      = extra.get("question_num", seg.question_num),
        example_num       = extra.get("example_num", seg.example_num),
        is_nomuna         = extra.get("is_nomuna", seg.is_nomuna),
        token_count       = estimate_tokens(seg.text),
    )


def build_chunks(
    segments: list[Segment],
    fig_lookup: dict[int, list[dict]],
) -> tuple[list[Chunk], list[Chunk]]:
    """Segments → (parent_chunks, child_chunks)"""
    print("[Phase 3] Building chunks...")

    parents:  list[Chunk] = []
    children: list[Chunk] = []

    # Section-level parent accumulator
    sec_parent: Optional[Chunk] = None
    sec_texts:  list[str] = []
    prev_section = ""

    def flush_section():
        nonlocal sec_parent, sec_texts
        if sec_parent and sec_texts:
            full = "\n\n".join(sec_texts)
            sec_parent.text        = full
            sec_parent.token_count = estimate_tokens(full)
            parents.append(sec_parent)
        sec_parent = None
        sec_texts  = []

    for seg in segments:

        # ── Fix 4: Junk page filter ──
        # cover page, TOC ইত্যাদি front matter skip করি
        # এগুলোতে chapter=0 এবং book_page=0 থাকে
        if seg.chapter_num == 0 and seg.book_page_start == 0:
            continue

        # Section change → flush section parent
        if seg.section_num != prev_section:
            flush_section()
            prev_section = seg.section_num

        # ── Structural headers ──
        if seg.content_type in ("section_header", "sub_section_header", "chapter_intro"):
            if seg.content_type == "section_header":
                flush_section()
                pid = make_chunk_id(f"ch{seg.chapter_num}_sec{seg.section_num}_parent")
                sec_parent = Chunk(
                    chunk_id=pid, parent_id=pid, chunk_type="parent",
                    content_type="theory", content_subtype="",
                    text="", chapter_num=seg.chapter_num, chapter_title=seg.chapter_title,
                    section_num=seg.section_num, section_title=seg.section_title,
                    sub_section_num="", book_page_start=seg.book_page_start,
                    book_page_end=seg.book_page_end, spans_pages=False,
                    onusondhan_id="", figure_id="", figure_image_path="",
                    table_id="", question_num=-1, example_num=-1,
                    is_nomuna=False,
                )
                sec_texts = [f"## {seg.section_num} {seg.section_title}"]
            continue

        # ── Figure ──
        if seg.content_type == "figure":
            img = match_figure_to_segment(seg, fig_lookup)
            parents.append(_make_parent(seg, extra={"figure_image_path": img}))
            continue

        # ── Complete Units (ভাগ ১) ──
        if seg.content_type in COMPLETE_UNIT_TYPES:
            flush_section()

            # Minimum token filter — "৬১" বা "" টাইপের junk parent skip
            MIN_COMPLETE_TOKENS = 5
            if estimate_tokens(seg.text) < MIN_COMPLETE_TOKENS:
                continue

            # Smart merge: একই onusondhan_id এর consecutive segments → একটা chunk
            if (
                seg.content_type == "onusondhan"
                and seg.onusondhan_id
                and parents
                and parents[-1].content_type == "onusondhan"
                and parents[-1].onusondhan_id == seg.onusondhan_id
            ):
                # Merge into the previous onusondhan chunk
                parents[-1].text += "\n\n" + seg.text
                parents[-1].book_page_end = max(parents[-1].book_page_end, seg.book_page_end)
                parents[-1].spans_pages   = (
                    parents[-1].book_page_start != parents[-1].book_page_end
                )
                parents[-1].token_count = estimate_tokens(parents[-1].text)
            else:
                parents.append(_make_parent(seg))
            continue


        # ── Table ──
        if seg.content_type == "table":
            parents.append(_make_parent(seg))
            continue

        # ── Exercise questions (ভাগ ২) ──
        if seg.content_type in EXERCISE_TYPES:
            if seg.content_subtype == "header":
                continue
            flush_section()
            parents.append(_make_parent(seg))
            continue

        # ── Example (standalone parent + section accumulation) ──
        if seg.content_type == "example":
            parents.append(_make_parent(seg))
            if sec_parent:
                sec_texts.append(seg.text)
            continue

        # ── Theory / Formula (ভাগ ৩: parent + children) ──
        if seg.content_type in ("theory", "formula"):
            if not sec_parent:
                pid = make_chunk_id(f"ch{seg.chapter_num}_theory")
                sec_parent = Chunk(
                    chunk_id=pid, parent_id=pid, chunk_type="parent",
                    content_type="theory", content_subtype="",
                    text="", chapter_num=seg.chapter_num, chapter_title=seg.chapter_title,
                    section_num=seg.section_num, section_title=seg.section_title,
                    sub_section_num=seg.sub_section_num, book_page_start=seg.book_page_start,
                    book_page_end=seg.book_page_end, spans_pages=False,
                    onusondhan_id="", figure_id="", figure_image_path="",
                    table_id="", question_num=-1, example_num=-1, is_nomuna=False,
                )
                sec_texts = []

            if seg.book_page_end > sec_parent.book_page_end:
                sec_parent.book_page_end = seg.book_page_end
                sec_parent.spans_pages = (sec_parent.book_page_start != sec_parent.book_page_end)

            sec_texts.append(seg.text)

            # Child chunks (paragraph-level)
            # Fix 5: minimum token threshold — ছোট junk child chunks বাদ দিই
            MIN_CHILD_TOKENS = 10   # ~30 characters — অর্থপূর্ণ sentence এর minimum
            for para in split_into_paragraphs(seg.text):
                if not para.strip():
                    continue
                tok = estimate_tokens(para)
                if tok < MIN_CHILD_TOKENS:
                    continue   # "E=mc2" বা "F=" টাইপের junk skip

                cid = make_chunk_id(f"ch{seg.chapter_num}_child")
                children.append(Chunk(
                    chunk_id=cid, parent_id=sec_parent.chunk_id,
                    chunk_type="child", content_type=seg.content_type,
                    content_subtype="", text=para,
                    chapter_num=seg.chapter_num, chapter_title=seg.chapter_title,
                    section_num=seg.section_num, section_title=seg.section_title,
                    sub_section_num=seg.sub_section_num,
                    book_page_start=seg.book_page_start, book_page_end=seg.book_page_end,
                    spans_pages=False, onusondhan_id="", figure_id="",
                    figure_image_path="", table_id="", question_num=-1,
                    example_num=-1, is_nomuna=False,
                    token_count=estimate_tokens(para),
                ))
            continue

    flush_section()
    print(f"  Done. {len(parents)} parents, {len(children)} children.")
    return parents, children


# ─────────────────────────────────────────────
# Phase 4: Save
# ─────────────────────────────────────────────
def save_jsonl(chunks: list[Chunk], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    print(f"  Saved {len(chunks)} chunks -> {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Physics Book — Chunker (v1)")
    print("=" * 60)

    if not MARKDOWN_FILE.exists():
        print(f"[ERROR] {MARKDOWN_FILE} not found!")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[0] Loading page map...")
    page_map = load_page_map()
    print(f"    {len(page_map)} pages mapped")

    fig_lookup = build_figure_lookup(FIGURE_JSON)

    segments = parse_markdown_to_segments(MARKDOWN_FILE, page_map)

    print("\n  Segment type distribution:")
    for ctype, count in Counter(s.content_type for s in segments).most_common():
        print(f"    {ctype:25s}: {count}")

    parents, children = build_chunks(segments, fig_lookup)

    print("\n[Phase 4] Saving...")
    save_jsonl(parents,  PARENT_CHUNKS)
    save_jsonl(children, CHILD_CHUNKS)

    print("\n" + "=" * 60)
    print("[DONE]")
    print(f"  Parent chunks : {len(parents)}")
    print(f"  Child chunks  : {len(children)}")
    print(f"  Total         : {len(parents) + len(children)}")
    print(f"\nNext: python scripts/3_embed_store.py")


if __name__ == "__main__":
    main()
