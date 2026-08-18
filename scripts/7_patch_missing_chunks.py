"""
scripts/7_patch_missing_chunks.py
===================================
PURPOSE:
    Chapter 2 এর end-of-chapter exercise sections
    (নমুনা প্রশ্ন, সংক্ষিপ্ত প্রশ্ন, সৃজনশীল প্রশ্ন)
    extracted_book.md থেকে পড়ে LanceDB তে add করা।

    এই sections চুনকারী আগে extract করেনি — সরাসরি markdown থেকে add করছি।
    Re-embed করতে হবে (Google text-embedding-004)।

    CHAPTER EXERCISE SECTIONS:
    Ch2: নমুনা প্রশ্ন → lines 2196-2252, book pages 59-60, ch=2
    Ch2: সৃজনশীল → lines 2254-2274, book pages 60-61, ch=2
    Ch2: সংক্ষিপ্ত → lines 2275-2279, book page 61, ch=2
"""

import io, sys, os, json, re, hashlib, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import lancedb
import numpy as np
from google import genai

GCP_PROJECT  = os.getenv("GCP_PROJECT", "project-3e580a5b-256c-4c6d-a0a")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
DB_PATH      = Path("data/lancedb")
BOOK_MD      = Path("data/processed/extracted_book.md")


def uid(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def embed(client, text: str) -> list[float]:
    result = client.models.embed_content(model=EMBED_MODEL, contents=[text])
    return list(result.embeddings[0].values)


# ─── Define all missing exercise chunks ─────────────────────
# Read the relevant lines from extracted_book.md

def read_book_lines(path: Path, start: int, end: int) -> str:
    """1-indexed, inclusive."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[start-1:end])


def get_missing_chunks() -> list[dict]:
    """
    Returns list of dicts with:
      chapter_num, section_num, content_type, book_page_start, book_page_end, text
    """
    chunks = []

    # ─── Chapter 2 ──────────────────────────────────────────
    # নমুনা প্রশ্ন (MCQ) — lines 2196-2252 → book pages 59-60
    mcq_text = read_book_lines(BOOK_MD, 2194, 2252)
    mcq_text_clean = re.sub(r"<!--.*?-->", "", mcq_text).strip()
    chunks.append({
        "chapter_num":    2,
        "section_num":    "2.end",
        "content_type":   "mcq_nomuna",
        "content_subtype": "question",
        "book_page_start": 59,
        "book_page_end":   60,
        "text":           "বহুনির্বাচনি নমুনা প্রশ্ন — অধ্যায় ২ (গতি)\n\n" + mcq_text_clean,
    })

    # সৃজনশীল প্রশ্ন — lines 2254-2274 → book pages 60-61
    sq_text = read_book_lines(BOOK_MD, 2253, 2274)
    sq_text_clean = re.sub(r"<!--.*?-->", "", sq_text).strip()
    chunks.append({
        "chapter_num":    2,
        "section_num":    "2.end",
        "content_type":   "creative_question",
        "content_subtype": "question",
        "book_page_start": 60,
        "book_page_end":   61,
        "text":           "সৃজনশীল প্রশ্ন — অধ্যায় ২ (গতি)\n\n" + sq_text_clean,
    })

    # সংক্ষিপ্ত উত্তর প্রশ্ন — lines 2275-2280 → book page 61
    sh_text = read_book_lines(BOOK_MD, 2275, 2280)
    sh_text_clean = re.sub(r"<!--.*?-->", "", sh_text).strip()
    chunks.append({
        "chapter_num":    2,
        "section_num":    "2.end",
        "content_type":   "shankhipto",
        "content_subtype": "question",
        "book_page_start": 61,
        "book_page_end":   61,
        "text":           "সংক্ষিপ্ত উত্তর প্রশ্ন — অধ্যায় ২ (গতি)\n\n" + sh_text_clean,
    })

    return chunks


def main():
    print("=== Patch: Missing Exercise Chunks ===\n")

    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    db     = lancedb.connect(str(DB_PATH))

    chunks = get_missing_chunks()
    print(f"Chunks to add: {len(chunks)}\n")

    for c in chunks:
        print(f"  Preview [{c['content_type']}] ch={c['chapter_num']} pg={c['book_page_start']}")
        print(f"  Text preview: {c['text'][:100]}")
        print()

    # Embed + build rows for parent_chunks
    parent_rows = []
    child_rows  = []

    pt = db.open_table("parent_chunks")
    ct = db.open_table("child_chunks")

    # Get sample schema row to know the field structure
    sample_p = pt.to_pandas().iloc[0]
    sample_c = ct.to_pandas().iloc[0]
    print("Parent schema columns:", list(sample_p.index))
    print("Child schema columns:", list(sample_c.index))
    print()

    for c in chunks:
        text = c["text"]
        vec  = embed(client, text)
        time.sleep(0.5)

        chunk_id  = f"patch_ch{c['chapter_num']}_{c['content_type']}_{uid(text)}"

        # Parent row (same as parent schema)
        prow = {}
        for col in sample_p.index:
            prow[col] = sample_p[col]  # default fill

        prow["vector"]           = vec
        prow["chunk_id"]         = chunk_id
        prow["parent_id"]        = chunk_id   # parent is itself
        prow["chunk_type"]       = "parent"
        prow["content_type"]     = c["content_type"]
        prow["content_subtype"]  = c.get("content_subtype", "")
        prow["text"]             = text
        prow["chapter_num"]      = c["chapter_num"]
        prow["chapter_title"]    = "গতি"
        prow["section_num"]      = c["section_num"]
        prow["section_title"]    = ""
        prow["sub_section_num"]  = ""
        prow["book_page_start"]  = c["book_page_start"]
        prow["book_page_end"]    = c["book_page_end"]
        prow["token_count"]      = len(text.split())
        parent_rows.append(prow)

        # Child row (smaller chunk = same text for now)
        crow = {}
        for col in sample_c.index:
            crow[col] = sample_c[col]

        crow["vector"]          = vec
        crow["chunk_id"]        = f"{chunk_id}_child"
        crow["parent_id"]       = chunk_id
        crow["chunk_type"]      = "child"
        crow["content_type"]    = c["content_type"]
        crow["content_subtype"] = c.get("content_subtype", "")
        crow["text"]            = text[:800]   # child is shorter
        crow["chapter_num"]     = c["chapter_num"]
        crow["chapter_title"]   = "গতি"
        crow["section_num"]     = c["section_num"]
        crow["section_title"]   = ""
        crow["sub_section_num"] = ""
        crow["book_page_start"] = c["book_page_start"]
        crow["book_page_end"]   = c["book_page_end"]
        crow["token_count"]     = len(text[:800].split())
        child_rows.append(crow)

        print(f"  Embedded: {chunk_id} ✓")

    # Add to tables
    import pandas as pd
    pt.add(pd.DataFrame(parent_rows))
    ct.add(pd.DataFrame(child_rows))

    print(f"\nAdded {len(parent_rows)} parent + {len(child_rows)} child chunks.")
    print(f"parent_chunks total: {pt.count_rows()}")
    print(f"child_chunks total:  {ct.count_rows()}")
    print("\nDone! ✓")


if __name__ == "__main__":
    main()
