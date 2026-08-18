"""
scripts/8_embed_exercise_sections.py
======================================
সব 13 chapter এর exercise sections (নমুনা প্রশ্ন, সংক্ষিপ্ত, সৃজনশীল, অনুসন্ধান)
BookIndex থেকে নিয়ে properly chunk করে embed করে LanceDB তে add করা।

Strategy:
  MCQ       → প্রতিটা question আলাদা child chunk, পুরো section একটা parent chunk
  Shankhipto→ একইভাবে
  Creative  → প্রতিটা creative question (ক/খ/গ/ঘ সহ) একটা chunk
  Onusondhan→ প্রতিটা অনুসন্ধান একটা chunk

Duplicate prevention: chunk_id prefix "bidx_" দিয়ে check করব।
"""

import io, sys, os, re, time, hashlib
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import lancedb
import numpy as np
import pandas as pd
from google import genai

GCP_PROJECT  = os.getenv("GCP_PROJECT", "project-3e580a5b-256c-4c6d-a0a")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
DB_PATH      = Path("data/lancedb")

CHAPTER_TITLES = {
    1:  "ভৌত রাশি এবং তাদের পরিমাপ",
    2:  "গতি",
    3:  "বল",
    4:  "কাজ, শক্তি ও ক্ষমতা",
    5:  "পদার্থের অবস্থা ও চাপ",
    6:  "তাপ",
    7:  "তরঙ্গ ও শব্দ",
    8:  "আলো",
    9:  "স্থির বিদ্যুৎ",
    10: "চল বিদ্যুৎ",
    11: "বিদ্যুতের চৌম্বকীয় ক্রিয়া ও চুম্বকত্ব",
    12: "আধুনিক পদার্থবিজ্ঞান ও ইলেকট্রনিক্স",
    13: "সেমিকন্ডাক্টর ও ইলেকট্রনিক্স",
}

BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# ─────────────────────────────────────────────────────────
# Question splitter
# ─────────────────────────────────────────────────────────
_Q_RE = re.compile(
    r"(?:^|\n)\s*([০-৯\d]+)[\.।]\s+",   # "১. " or "1. " at start of line
)

def split_questions(text: str) -> list[tuple[int, str]]:
    """
    text কে numbered questions এ split করো।
    Return: [(question_num, question_text), ...]
    """
    matches = list(_Q_RE.finditer(text))
    if len(matches) < 2:
        return [(0, text.strip())]

    results = []
    for i, m in enumerate(matches):
        raw_num = m.group(1).translate(BN2EN)
        q_num   = int(raw_num)
        start   = m.start()
        end     = matches[i+1].start() if i+1 < len(matches) else len(text)
        q_text  = text[start:end].strip()
        if len(q_text) > 10:
            results.append((q_num, q_text))
    return results


def split_creative(text: str) -> list[tuple[int, str]]:
    """সৃজনশীল questions split — each numbered block is one question."""
    # Creative questions start with ১. or 1. followed by a scenario
    return split_questions(text)


# ─────────────────────────────────────────────────────────
# Embed helper
# ─────────────────────────────────────────────────────────
def embed_text(client, text: str, retries: int = 3) -> list[float]:
    for attempt in range(retries):
        try:
            result = client.models.embed_content(model=EMBED_MODEL, contents=[text])
            return list(result.embeddings[0].values)
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt+1}: {e}")
                time.sleep(2 ** attempt)
            else:
                raise


def uid(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:6]


# ─────────────────────────────────────────────────────────
# Row builder
# ─────────────────────────────────────────────────────────
def make_parent_row(sample, chunk_id, vec, text, ch, ctype, q_num=None, on_id=None):
    r = {col: sample[col] for col in sample.index}
    r["vector"]           = vec
    r["chunk_id"]         = chunk_id
    r["parent_id"]        = chunk_id
    r["chunk_type"]       = "parent"
    r["content_type"]     = ctype
    r["content_subtype"]  = "question"
    r["text"]             = text
    r["chapter_num"]      = ch
    r["chapter_title"]    = CHAPTER_TITLES.get(ch, "")
    r["section_num"]      = f"{ch}.end"
    r["section_title"]    = ""
    r["sub_section_num"]  = ""
    r["book_page_start"]  = 0
    r["book_page_end"]    = 0
    r["token_count"]      = len(text.split())
    r["question_num"]     = q_num or 0
    r["onusondhan_id"]    = on_id or ""
    r["is_nomuna"]        = (ctype == "mcq_nomuna")
    r["spans_pages"]      = False
    r["figure_id"]        = ""
    r["figure_image_path"]= ""
    r["table_id"]         = ""
    r["example_num"]      = 0
    return r


def make_child_row(sample, chunk_id, parent_id, vec, text, ch, ctype):
    r = {col: sample[col] for col in sample.index}
    r["vector"]          = vec
    r["chunk_id"]        = chunk_id
    r["parent_id"]       = parent_id
    r["chunk_type"]      = "child"
    r["content_type"]    = ctype
    r["content_subtype"] = "question"
    r["text"]            = text[:600]
    r["chapter_num"]     = ch
    r["chapter_title"]   = CHAPTER_TITLES.get(ch, "")
    r["section_num"]     = f"{ch}.end"
    r["section_title"]   = ""
    r["sub_section_num"] = ""
    r["book_page_start"] = 0
    r["book_page_end"]   = 0
    r["token_count"]     = len(text[:600].split())
    return r


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("Embed Exercise Sections → LanceDB (all 13 chapters)")
    print("=" * 68)

    client   = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    db       = lancedb.connect(str(DB_PATH))
    pt       = db.open_table("parent_chunks")
    ct       = db.open_table("child_chunks")

    sample_p = pt.to_pandas().iloc[0]
    sample_c = ct.to_pandas().iloc[0]

    # Existing chunk IDs (to avoid duplicates)
    existing_ids = set(pt.to_pandas()["chunk_id"].tolist())
    print(f"Existing parent chunks: {len(existing_ids)}")

    from src.book_index import get_book_index
    bidx = get_book_index()

    all_p_rows = []
    all_c_rows = []
    total_new  = 0

    for ch in range(1, 14):
        ex = bidx.get_chapter(ch)
        if not ex:
            print(f"\nCh {ch}: no data in BookIndex")
            continue

        print(f"\n{'─'*50}")
        print(f"Chapter {ch}: {CHAPTER_TITLES.get(ch, '')}")

        # ── MCQ / নমুনা প্রশ্ন ────────────────────────────
        if ex.mcq_text:
            questions = split_questions(ex.mcq_text)
            parent_id = f"bidx_mcq_ch{ch}"

            # Parent chunk = whole MCQ section
            if parent_id not in existing_ids:
                print(f"  MCQ: embedding parent ({len(ex.mcq_text)} chars)...")
                vec = embed_text(client, ex.mcq_text)
                time.sleep(0.3)
                all_p_rows.append(make_parent_row(
                    sample_p, parent_id, vec, ex.mcq_text, ch, "mcq_nomuna"
                ))
                existing_ids.add(parent_id)
                total_new += 1
            else:
                print(f"  MCQ parent: already exists, skipping")

            # Child chunks = individual questions
            for q_num, q_text in questions:
                cid = f"bidx_mcq_ch{ch}_q{q_num}_{uid(q_text)}"
                if cid not in existing_ids:
                    vec_c = embed_text(client, q_text)
                    time.sleep(0.2)
                    all_c_rows.append(make_child_row(
                        sample_c, cid, parent_id, vec_c, q_text, ch, "mcq_nomuna"
                    ))
                    existing_ids.add(cid)
                    total_new += 1
                    print(f"    Q{q_num}: embedded ({len(q_text)} chars)")

        # ── সংক্ষিপ্ত উত্তর প্রশ্ন ───────────────────────
        if ex.shankhipto_text:
            questions = split_questions(ex.shankhipto_text)
            parent_id = f"bidx_shankhipto_ch{ch}"

            if parent_id not in existing_ids:
                print(f"  Shankhipto: embedding parent...")
                vec = embed_text(client, ex.shankhipto_text)
                time.sleep(0.3)
                all_p_rows.append(make_parent_row(
                    sample_p, parent_id, vec, ex.shankhipto_text, ch, "shankhipto"
                ))
                existing_ids.add(parent_id)
                total_new += 1

            for q_num, q_text in questions:
                cid = f"bidx_sh_ch{ch}_q{q_num}_{uid(q_text)}"
                if cid not in existing_ids:
                    vec_c = embed_text(client, q_text)
                    time.sleep(0.2)
                    all_c_rows.append(make_child_row(
                        sample_c, cid, parent_id, vec_c, q_text, ch, "shankhipto"
                    ))
                    existing_ids.add(cid)
                    total_new += 1
                    print(f"    Sh Q{q_num}: embedded")

        # ── সৃজনশীল প্রশ্ন ────────────────────────────────
        if ex.creative_text:
            questions = split_creative(ex.creative_text)
            parent_id = f"bidx_creative_ch{ch}"

            if parent_id not in existing_ids:
                print(f"  Creative: embedding parent...")
                vec = embed_text(client, ex.creative_text)
                time.sleep(0.3)
                all_p_rows.append(make_parent_row(
                    sample_p, parent_id, vec, ex.creative_text, ch, "creative_question"
                ))
                existing_ids.add(parent_id)
                total_new += 1

            for q_num, q_text in questions:
                cid = f"bidx_cr_ch{ch}_q{q_num}_{uid(q_text)}"
                if cid not in existing_ids:
                    vec_c = embed_text(client, q_text)
                    time.sleep(0.2)
                    all_c_rows.append(make_child_row(
                        sample_c, cid, parent_id, vec_c, q_text, ch, "creative_question"
                    ))
                    existing_ids.add(cid)
                    total_new += 1
                    print(f"    Cr Q{q_num}: embedded")

        # ── অনুসন্ধান ──────────────────────────────────────
        for entry in ex.onusondhan_texts:
            on_id     = entry["id"]
            on_text   = entry["text"]
            parent_id = f"bidx_onusondan_ch{ch}_{on_id}"

            if parent_id not in existing_ids:
                print(f"  Onusondhan {on_id}: embedding...")
                vec = embed_text(client, on_text)
                time.sleep(0.3)
                r   = make_parent_row(
                    sample_p, parent_id, vec, on_text, ch, "onusondhan", on_id=on_id
                )
                r["book_page_start"] = entry.get("book_page", 0)
                r["book_page_end"]   = entry.get("book_page", 0) + 2
                all_p_rows.append(r)
                existing_ids.add(parent_id)
                total_new += 1
                print(f"    Onusondhan {on_id}: embedded")

    # ── Batch write ─────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"New rows to add: {total_new} "
          f"({len(all_p_rows)} parent + {len(all_c_rows)} child)")

    if all_p_rows:
        pt.add(pd.DataFrame(all_p_rows))
        print(f"parent_chunks total: {pt.count_rows()}")

    if all_c_rows:
        ct.add(pd.DataFrame(all_c_rows))
        print(f"child_chunks total:  {ct.count_rows()}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
