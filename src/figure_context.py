"""
src/figure_context.py
======================
Page-based figure description lookup।

যখন retrieval কোনো page/section এর content আনে,
সেই page range এর figure descriptions automatically context এ যোগ করো —
যাতে LLM চিত্র দেখাতে না পারলেও describe করতে পারে।

USAGE:
    from src.figure_context import get_figures_for_pages, figures_to_context

    figs = get_figures_for_pages(43, 46)
    fig_text = figures_to_context(figs)
"""

import json
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────
# Lazy load cache
# ─────────────────────────────────────────────────────────
_ALL_FIGS: Optional[dict]       = None  # key → fig_dict
_BY_PAGE:  Optional[dict[int, list]] = None  # book_page → [fig_dict, ...]


def _load() -> None:
    global _ALL_FIGS, _BY_PAGE
    if _ALL_FIGS is not None:
        return

    fig_path = Path("data/processed/figure_descriptions.json")
    if not fig_path.exists():
        _ALL_FIGS = {}
        _BY_PAGE  = {}
        return

    with open(fig_path, encoding="utf-8") as f:
        _ALL_FIGS = json.load(f)

    # Build page index
    _BY_PAGE = {}
    for key, fig in _ALL_FIGS.items():
        pg = fig.get("book_page")
        if pg is not None:
            _BY_PAGE.setdefault(int(pg), []).append(fig)


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def get_figures_for_pages(
    page_start: int,
    page_end:   int,
    max_figs:   int = 4,
) -> list[dict]:
    """
    page_start থেকে page_end পর্যন্ত range এর figures return করো।
    Description আছে এমন figures-ই return করা হবে।
    """
    _load()
    results = []
    for pg in range(page_start, page_end + 1):
        for fig in _BY_PAGE.get(pg, []):
            if fig.get("description", "").strip():
                results.append(fig)
    return results[:max_figs]


def get_figures_for_chapter(
    chapter_num: int,
    max_figs:    int = 8,
) -> list[dict]:
    """একটা chapter এর সব figures return করো।"""
    _load()
    results = []
    for fig in _ALL_FIGS.values():
        if fig.get("chapter_num") == chapter_num and fig.get("description", "").strip():
            results.append(fig)
    # Sort by book_page
    results.sort(key=lambda x: (x.get("book_page", 0), x.get("index_on_page", 0)))
    return results[:max_figs]


def figures_to_context(figs: list[dict]) -> str:
    """
    Figure descriptions থেকে LLM-ready context text বানাও।
    চিত্রের বর্ণনা বাংলায় আছে, সেটাই pass করো।
    """
    if not figs:
        return ""

    parts = []
    for fig in figs:
        pg      = fig.get("book_page", "?")
        cap     = fig.get("caption", "").strip()
        fig_id  = fig.get("figure_id", "").strip()
        desc    = fig.get("description", "").strip()
        ch      = fig.get("chapter_num", "?")

        if not desc:
            continue

        # Header
        header_parts = [f"[চিত্র | অধ্যায় {ch} | পৃষ্ঠা {pg}]"]
        if fig_id:
            header_parts.append(f"চিত্র {fig_id}")
        if cap:
            header_parts.append(cap)
        header = " — ".join(header_parts)

        parts.append(f"{header}\n{desc[:600]}")

    return "\n\n".join(parts) if parts else ""


def inject_figures_into_context(
    existing_context: str,
    results:          list[dict],
    max_total_figs:   int = 4,
) -> str:
    """
    Retrieval results এর page range থেকে figures inject করো।
    Already-built context এর শেষে figure section যোগ করো।
    """
    if not results:
        return existing_context

    # Collect page ranges from results
    all_figs: list[dict] = []
    seen_keys: set[str]  = set()

    for r in results:
        ps = r.get("page_start", 0)
        pe = r.get("page_end",   ps)
        for fig in get_figures_for_pages(ps, pe, max_figs=2):
            key = fig.get("key", "")
            if key not in seen_keys:
                seen_keys.add(key)
                all_figs.append(fig)

    all_figs = all_figs[:max_total_figs]
    if not all_figs:
        return existing_context

    fig_text = figures_to_context(all_figs)
    if not fig_text:
        return existing_context

    return existing_context + "\n\n---\n\n**[সংশ্লিষ্ট চিত্রের বর্ণনা]**\n\n" + fig_text
