"""
scripts/3_embed_store.py
========================
PURPOSE:
    parent_chunks.jsonl + child_chunks.jsonl
    → Vertex AI text-embedding-004 দিয়ে embed
    → LanceDB তে দুটো table এ store

TABLES:
    child_chunks  — vector search এর জন্য (ছোট paragraphs)
    parent_chunks — context retrieval এর জন্য (বড় sections)

FEATURES:
    - Batch embedding (100 chunks/request)
    - Resume support (মাঝপথে থামলে আবার শুরু থেকে না)
    - Exponential backoff on rate limit
    - Progress tracking

USAGE:
    python scripts/3_embed_store.py
"""

import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

import lancedb
import pyarrow as pa
from google import genai

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
GCP_PROJECT  = os.getenv("GCP_PROJECT", "project-3e580a5b-256c-4c6d-a0a")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

PARENT_JSONL = Path("data/processed/parent_chunks.jsonl")
CHILD_JSONL  = Path("data/processed/child_chunks.jsonl")
DB_PATH      = Path("data/lancedb")

BATCH_SIZE      = 100    # child chunks এর জন্য (ছোট, safe)
# Parent chunks বড় হয় — token-aware batching use করব
# text-embedding-004 limit: 20,000 tokens per request
# 1 token ~ 3 chars (Bengali), safe margin এ 15,000 tokens = 45,000 chars
MAX_BATCH_CHARS = 45_000
EMBED_DIM       = 768    # text-embedding-004 output dimension

RETRY_DELAYS = [2, 5, 15, 30, 60]  # Rate limit hit হলে এতক্ষণ wait


# ─────────────────────────────────────────────
# LanceDB Schema
# ─────────────────────────────────────────────
CHILD_SCHEMA = pa.schema([
    pa.field("vector",          pa.list_(pa.float32(), EMBED_DIM)),
    pa.field("chunk_id",        pa.utf8()),
    pa.field("parent_id",       pa.utf8()),
    pa.field("chunk_type",      pa.utf8()),
    pa.field("content_type",    pa.utf8()),
    pa.field("content_subtype", pa.utf8()),
    pa.field("text",            pa.utf8()),
    pa.field("chapter_num",     pa.int32()),
    pa.field("chapter_title",   pa.utf8()),
    pa.field("section_num",     pa.utf8()),
    pa.field("section_title",   pa.utf8()),
    pa.field("sub_section_num", pa.utf8()),
    pa.field("book_page_start", pa.int32()),
    pa.field("book_page_end",   pa.int32()),
    pa.field("token_count",     pa.int32()),
])

PARENT_SCHEMA = pa.schema([
    pa.field("vector",            pa.list_(pa.float32(), EMBED_DIM)),
    pa.field("chunk_id",          pa.utf8()),
    pa.field("parent_id",         pa.utf8()),
    pa.field("chunk_type",        pa.utf8()),
    pa.field("content_type",      pa.utf8()),
    pa.field("content_subtype",   pa.utf8()),
    pa.field("text",              pa.utf8()),
    pa.field("chapter_num",       pa.int32()),
    pa.field("chapter_title",     pa.utf8()),
    pa.field("section_num",       pa.utf8()),
    pa.field("section_title",     pa.utf8()),
    pa.field("sub_section_num",   pa.utf8()),
    pa.field("book_page_start",   pa.int32()),
    pa.field("book_page_end",     pa.int32()),
    pa.field("spans_pages",       pa.bool_()),
    pa.field("onusondhan_id",     pa.utf8()),
    pa.field("figure_id",         pa.utf8()),
    pa.field("figure_image_path", pa.utf8()),
    pa.field("table_id",          pa.utf8()),
    pa.field("question_num",      pa.int32()),
    pa.field("example_num",       pa.int32()),
    pa.field("is_nomuna",         pa.bool_()),
    pa.field("token_count",       pa.int32()),
])


# ─────────────────────────────────────────────
# Embedding with retry
# ─────────────────────────────────────────────
def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """texts → 768-dim vectors. Rate limit হলে retry করে।"""
    for attempt, delay in enumerate(RETRY_DELAYS + [None]):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
            )
            return [list(e.values) for e in result.embeddings]

        except Exception as e:
            err = str(e)
            if delay is None:
                raise RuntimeError(f"All retries failed: {e}") from e
            if any(x in err for x in ["RATE_LIMIT", "429", "quota", "UNAVAILABLE", "503"]):
                print(f"    [Retry {attempt+1}] waiting {delay}s... ({err[:60]})")
                time.sleep(delay)
            else:
                raise  # Unknown error — don't retry


# ─────────────────────────────────────────────
# Load chunks
# ─────────────────────────────────────────────
def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  {len(chunks)} chunks loaded from {path.name}")
    return chunks


# ─────────────────────────────────────────────
# Build a row dict for LanceDB
# ─────────────────────────────────────────────
def make_child_row(chunk: dict, vec: list[float]) -> dict:
    return {
        "vector":          vec,
        "chunk_id":        chunk["chunk_id"],
        "parent_id":       chunk["parent_id"],
        "chunk_type":      chunk["chunk_type"],
        "content_type":    chunk["content_type"],
        "content_subtype": chunk.get("content_subtype", ""),
        "text":            chunk["text"],
        "chapter_num":     int(chunk["chapter_num"]),
        "chapter_title":   chunk.get("chapter_title", ""),
        "section_num":     chunk.get("section_num", ""),
        "section_title":   chunk.get("section_title", ""),
        "sub_section_num": chunk.get("sub_section_num", ""),
        "book_page_start": int(chunk["book_page_start"]),
        "book_page_end":   int(chunk["book_page_end"]),
        "token_count":     int(chunk.get("token_count", 0)),
    }


def make_parent_row(chunk: dict, vec: list[float]) -> dict:
    return {
        "vector":            vec,
        "chunk_id":          chunk["chunk_id"],
        "parent_id":         chunk["parent_id"],
        "chunk_type":        chunk["chunk_type"],
        "content_type":      chunk["content_type"],
        "content_subtype":   chunk.get("content_subtype", ""),
        "text":              chunk["text"],
        "chapter_num":       int(chunk["chapter_num"]),
        "chapter_title":     chunk.get("chapter_title", ""),
        "section_num":       chunk.get("section_num", ""),
        "section_title":     chunk.get("section_title", ""),
        "sub_section_num":   chunk.get("sub_section_num", ""),
        "book_page_start":   int(chunk["book_page_start"]),
        "book_page_end":     int(chunk["book_page_end"]),
        "spans_pages":       bool(chunk.get("spans_pages", False)),
        "onusondhan_id":     chunk.get("onusondhan_id", ""),
        "figure_id":         chunk.get("figure_id", ""),
        "figure_image_path": chunk.get("figure_image_path", ""),
        "table_id":          chunk.get("table_id", ""),
        "question_num":      int(chunk.get("question_num", -1)),
        "example_num":       int(chunk.get("example_num", -1)),
        "is_nomuna":         bool(chunk.get("is_nomuna", False)),
        "token_count":       int(chunk.get("token_count", 0)),
    }

def make_batches(chunks: list[dict], max_chars: int = MAX_BATCH_CHARS) -> list[list[dict]]:
    """
    Token-aware dynamic batching.
    প্রতিটা batch এ max_chars এর বেশি character জমা হলে নতুন batch শুরু হয়।
    এতে Vertex AI এর 20,000 token limit এ ধাক্কা খাওয়া যাবে না।
    """
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_chars = 0

    for chunk in chunks:
        text_len = len(chunk["text"])
        # বর্তমান batch এ এটা add করলে limit exceed হলে নতুন batch শুরু
        if current_batch and (current_chars + text_len > max_chars):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(chunk)
        current_chars += text_len

    if current_batch:
        batches.append(current_batch)

    return batches



# ─────────────────────────────────────────────
# Embed and Store
# ─────────────────────────────────────────────
def embed_and_store(
    client:     genai.Client,
    db:         lancedb.DBConnection,
    chunks:     list[dict],
    table_name: str,
    schema:     pa.Schema,
    row_fn,                  # make_child_row or make_parent_row
):
    print(f"\n  [{table_name}] processing {len(chunks)} chunks...")

    # Resume: table already সম্পূর্ণ থাকলে skip
    # (count_rows চেক — to_pandas এর চেয়ে reliable)
    try:
        existing      = db.open_table(table_name)
        stored_count  = existing.count_rows()
        total_needed  = len(chunks)

        if stored_count >= total_needed:
            print(f"  Already complete ({stored_count} rows). Skipping!")
            return
        elif stored_count > 0:
            print(f"  Partial: {stored_count}/{total_needed} stored. Dropping & re-embedding...")
            db.drop_table(table_name)
    except Exception:
        pass  # table নেই — fresh start

    to_embed = chunks  # সবসময় সব embed করি (partial হলে drop করেছি)
    print(f"  To embed: {len(to_embed)} chunks")

    # Open or create table
    try:
        table = db.create_table(table_name, schema=schema)
        print(f"  Created table '{table_name}'")
    except ValueError:
        table = db.open_table(table_name)
        print(f"  Opened existing table '{table_name}'")


    # Token-aware batch তৈরি
    batches   = make_batches(to_embed)
    total     = len(to_embed)
    stored_count = 0
    t_start   = time.time()
    print(f"  Batches: {len(batches)} (token-aware, max {MAX_BATCH_CHARS} chars/batch)")

    for b_idx, batch in enumerate(batches):
        texts = [c["text"] for c in batch]

        vectors = embed_batch(client, texts)

        rows = [row_fn(c, v) for c, v in zip(batch, vectors)]
        table.add(rows)
        stored_count += len(rows)

        # Progress
        elapsed  = time.time() - t_start
        pct      = stored_count / total * 100
        rate     = stored_count / elapsed if elapsed > 0 else 1
        eta_min  = (total - stored_count) / rate / 60
        batch_chars = sum(len(c["text"]) for c in batch)
        print(
            f"  [{pct:5.1f}%] {stored_count}/{total}  "
            f"| batch={len(batch)} ({batch_chars//1000}k chars)  "
            f"| {rate:.1f} ch/s  | ETA ~{eta_min:.1f}m"
        )

        time.sleep(0.3)  # gentle pacing

    # ANN index — fast cosine search এর জন্য
    print(f"  Building ANN index...")
    try:
        import lancedb.index as ldb_index
        table.create_index(
            "vector",
            config=ldb_index.IvfPq(
                distance_type="cosine",
                num_partitions=min(32, max(1, stored_count // 100)),
                num_sub_vectors=16,
            ),
        )
        print(f"  ANN index ready!")
    except Exception as e:
        print(f"  [WARN] ANN index build skipped (will use brute-force search): {e}")

    # FTS index — Bengali keyword search এর জন্য (hybrid retrieval)
    # Bengali-optimized: no stemming (English stemmer breaks Bengali words),
    # no stop word removal (English stop list not applicable to Bengali)
    if chunk_type == "parent":   # FTS only on parent_chunks (larger, more complete text)
        print(f"  Building FTS index on 'text' column (Bengali hybrid search)...")
        try:
            from lancedb.index import FTS
            table.create_index(
                "text",
                config=FTS(
                    base_tokenizer="simple",
                    language="English",       # simple space tokenizer — correct for Bengali
                    lower_case=False,         # Bengali has no case
                    stem=False,               # English stemmer destroys Bengali words
                    remove_stop_words=False,  # English stop words != Bengali
                    with_position=True,       # enables phrase search
                ),
                replace=True,
            )
            print(f"  FTS index ready!")
        except Exception as e:
            print(f"  [WARN] FTS index build skipped: {e}")



# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Physics Book -- Embed & Store")
    print("=" * 60)

    for f in [PARENT_JSONL, CHILD_JSONL]:
        if not f.exists():
            print(f"[ERROR] {f} not found. Run 2_chunk.py first.")
            sys.exit(1)

    DB_PATH.mkdir(parents=True, exist_ok=True)

    # Vertex AI client
    print(f"\n[1] Vertex AI setup...")
    print(f"    project : {GCP_PROJECT}")
    print(f"    location: {GCP_LOCATION}")
    print(f"    model   : {EMBED_MODEL}")
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    print("    OK")

    # LanceDB
    print(f"\n[2] LanceDB: {DB_PATH}")
    db = lancedb.connect(str(DB_PATH))
    print(f"    Existing tables: {db.list_tables()}")

    # Load
    print(f"\n[3] Loading chunks...")
    parent_chunks = load_chunks(PARENT_JSONL)
    child_chunks  = load_chunks(CHILD_JSONL)
    total         = len(parent_chunks) + len(child_chunks)
    api_calls     = total // BATCH_SIZE + 1
    print(f"    Total: {total} chunks (~{api_calls} API calls)")

    # Embed child chunks (smaller, used for search)
    print(f"\n[4] child_chunks...")
    embed_and_store(client, db, child_chunks, "child_chunks", CHILD_SCHEMA, make_child_row)

    # Embed parent chunks (larger, used for context)
    print(f"\n[5] parent_chunks...")
    embed_and_store(client, db, parent_chunks, "parent_chunks", PARENT_SCHEMA, make_parent_row)

    # Final summary
    print("\n" + "=" * 60)
    print("[DONE]")
    try:
        tables = db.table_names()
    except Exception:
        tables = []
    for tname in tables:
        t = db.open_table(tname)
        print(f"  {tname}: {t.count_rows()} rows stored")
    print(f"\nNext: python scripts/4_query_test.py")


if __name__ == "__main__":
    main()
