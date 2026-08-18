"""
scripts/4_query_test.py
========================
PURPOSE:
    LanceDB তে stored embeddings test করা।
    Real queries দিয়ে দেখব:
      1. সঠিক chapter/section আসছে কিনা
      2. ধরন অনুযায়ী content আসছে কিনা
      3. Parent-child fetch কাজ করছে কিনা

USAGE:
    python scripts/4_query_test.py
"""

import io
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import lancedb
from google import genai

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
GCP_PROJECT  = os.getenv("GCP_PROJECT", "project-3e580a5b-256c-4c6d-a0a")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
DB_PATH      = Path("data/lancedb")


def embed_query(client: genai.Client, query: str) -> list[float]:
    result = client.models.embed_content(model=EMBED_MODEL, contents=[query])
    return list(result.embeddings[0].values)


# Bengali প্রশ্নের common শব্দ — এগুলো content word না
_BN_STOP = {
    "কী", "কি", "কোনটি", "কোনটা", "কাকে", "বলে", "কেন", "কীভাবে",
    "হয়", "ব্যাখ্যা", "করো", "করো।", "নির্ণয়", "হলে", "পরে", "কত",
    "কোথায়", "কখন", "কারণ", "ও", "এবং", "বা", "থেকে", "এর",
    "তার", "সে", "যে", "এই", "ওই", "একটি", "একটা", "একটু",
    "দিয়ে", "নিয়ে", "কোনো", "সব", "আছে", "করে", "হলো", "হবে",
    "দিয়ে", "সূত্রটি", "সূত্রটা", "কিসের", "কিসের?", "কত?",
    "কী?", "কি?", "কোনটি?", "কাকে?", "কীভাবে?",
}


def _strip_bn_suffix(word: str) -> str:
    """
    Bengali inflection suffix strip করো।
    "ত্বরণের" → "ত্বরণ", "বেগে" → "বেগ", "আধানটি" → "আধান"
    """
    # দীর্ঘ suffix আগে check করি (short এর আগে long match হওয়া দরকার)
    suffixes = ["ের", "এর", "টির", "টার", "দের", "টিকে", "টাকে",
                "কে", "টি", "টা", "তে", "তেই", "য়ে", "য়"]
    for suf in suffixes:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return word


def extract_keywords(query: str) -> list[str]:
    """
    Query থেকে content keywords বের করো।
    প্রশ্নের শব্দ (কী, কোনটি, কেন...) বাদ দিয়ে
    বাকি meaningful শব্দগুলো return করে।
    Inflected form + base form দুটোই return করে।
    """
    import re
    clean = re.sub(r"[।?!,;:।\"'()\[\]]", " ", query)
    words = clean.split()
    keywords: list[str] = []
    for w in words:
        w = w.strip()
        if len(w) < 3 or w in _BN_STOP:
            continue
        keywords.append(w)                # original form: "ত্বরণের"
        stripped = _strip_bn_suffix(w)
        if stripped != w and len(stripped) >= 2:
            keywords.append(stripped)     # base form: "ত্বরণ"
    return list(dict.fromkeys(keywords))  # duplicates সরাই, order রাখি



def search_and_fetch(
    client:       genai.Client,
    child_table,
    parent_table,
    child_df,      # cached pandas dataframe + vectors
    child_vecs,    # cached numpy array of vectors
    parent_lookup: dict,  # parent_id → parent text (preloaded)
    query:        str,
    top_k:        int = 5,
) -> list[dict]:
    """
    3-layer retrieval:
      1. query embed করো
      2. exact cosine similarity (top 20 candidates)
      3. keyword boost দিয়ে re-rank → top_k
      4. parent fetch
    """
    import numpy as np

    # Step 1: embed query
    q_vec    = embed_query(client, query)
    q_arr    = np.array(q_vec, dtype=np.float32)
    keywords = extract_keywords(query)

    # Step 2: exact cosine similarity → top 20 candidates
    CANDIDATE_K = min(20, len(child_df))
    dots     = child_vecs @ q_arr
    norms    = np.linalg.norm(child_vecs, axis=1) * np.linalg.norm(q_arr)
    sims     = dots / (norms + 1e-10)
    cand_idx = np.argsort(-sims)[:CANDIDATE_K]

    # Step 3: keyword boost re-ranking
    # child text + parent text দুটোতেই keyword match করি
    CHILD_BOOST  = 0.08   # child text এ keyword থাকলে
    PARENT_BOOST = 0.05   # parent text এ keyword থাকলে (কম boost)
    final_scores = []
    for idx in cand_idx:
        base_score  = float(sims[idx])
        child_text  = str(child_df.iloc[idx]["text"])
        pid         = str(child_df.iloc[idx]["parent_id"])
        parent_text = parent_lookup.get(pid, "")

        child_bonus  = sum(CHILD_BOOST  for kw in keywords if kw in child_text)
        parent_bonus = sum(PARENT_BOOST for kw in keywords if kw in parent_text)
        final_scores.append((idx, base_score + child_bonus + parent_bonus))

    # descending sort by boosted score
    final_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in final_scores[:top_k]]
    top_scores  = [sc  for _,  sc in final_scores[:top_k]]

    hits   = child_df.iloc[top_indices].to_dict("records")
    scores = top_scores

    # Step 3: fetch parent for each unique parent_id
    seen_parents: set[str] = set()
    results = []

    for hit, score in zip(hits, scores):
        pid = hit["parent_id"]

        parent_text         = ""
        parent_content_type = ""
        if pid not in seen_parents:
            seen_parents.add(pid)
            parent_rows = (
                parent_table.search()
                .where(f"chunk_id = '{pid}'")
                .limit(1)
                .select(["chunk_id", "content_type", "text",
                         "chapter_num", "chapter_title", "section_num",
                         "section_title", "book_page_start", "book_page_end",
                         "onusondhan_id", "figure_image_path", "token_count"])
                .to_list()
            )
            if parent_rows:
                p = parent_rows[0]
                parent_text         = p["text"]
                parent_content_type = p["content_type"]

        results.append({
            "query":         query,
            "score":         float(score),
            "child_id":      hit["chunk_id"],
            "parent_id":     pid,
            "child_type":    hit["content_type"],
            "parent_type":   parent_content_type,
            "chapter":       hit["chapter_num"],
            "section":       hit["section_num"],
            "page":          hit["book_page_start"],
            "child_text":    str(hit["text"])[:200],
            "parent_text":   parent_text[:300] if parent_text else "(same as child)",
            "parent_tokens": len(parent_text) // 3 if parent_text else 0,
        })

    return results



def print_results(results: list[dict], label: str):
    print(f"\n{'='*65}")
    print(f"Query: {label}")
    print(f"{'='*65}")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] score={r['score']:.4f}  "
              f"ch={r['chapter']}  sec={r['section']}  pg={r['page']}")
        print(f"       child_type={r['child_type']} → parent_type={r['parent_type']}")
        print(f"       child : {repr(r['child_text'][:120])}")
        print(f"       parent: {repr(r['parent_text'][:150])}  "
              f"[{r['parent_tokens']} tok]")


# ─────────────────────────────────────────────
# Test queries
# ─────────────────────────────────────────────
TEST_QUERIES = [
    # তত্ত্ব সংক্রান্ত
    "নিউটনের গতির দ্বিতীয় সূত্র কী?",
    "ত্বরণ কাকে বলে এবং এর একক কী?",

    # সূত্র সংক্রান্ত
    "v = u + at সূত্রটি কিসের?",

    # অনুসন্ধান সংক্রান্ত
    "স্লাইড ক্যালিপার্স দিয়ে কীভাবে পরিমাপ করতে হয়?",

    # MCQ টাইপ
    "ত্বরণের একক কোনটি?",

    # Chapter specific
    "তড়িৎ আধান কী?",

    # সংখ্যাগত সমস্যা
    "একটি গাড়ি ৫ m/s বেগে চলছে, ত্বরণ ২ m/s² হলে ১০ সেকেন্ড পরে বেগ কত?",
]


def main():
    print("=" * 65)
    print("Physics Tutor — Query Test")
    print("=" * 65)

    # Connect
    print(f"\n[1] Connecting...")
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    db     = lancedb.connect(str(DB_PATH))

    tables = db.table_names()
    print(f"    Tables: {tables}")

    child_table  = db.open_table("child_chunks")
    parent_table = db.open_table("parent_chunks")
    print(f"    child_chunks : {child_table.count_rows()} rows")
    print(f"    parent_chunks: {parent_table.count_rows()} rows")

    # numpy cache: child table একবারে load করি (সব query এর জন্য)
    print(f"\n[2] Loading child vectors into memory for exact search...")
    import numpy as np
    child_df   = child_table.to_pandas()
    child_vecs = np.array(child_df["vector"].tolist(), dtype=np.float32)
    print(f"    child_vecs shape: {child_vecs.shape}")

    # parent text preload — keyword boost এর জন্য (vector ছাড়া)
    print(f"    Loading parent texts for keyword matching...")
    parent_df     = parent_table.search().select(["chunk_id", "text"]).limit(10000).to_pandas()
    parent_lookup = dict(zip(parent_df["chunk_id"], parent_df["text"]))
    print(f"    parent_lookup: {len(parent_lookup)} entries")

    # Run test queries
    print(f"\n[3] Running {len(TEST_QUERIES)} test queries...\n")

    total_time = 0
    for query in TEST_QUERIES:
        t0      = time.time()
        results = search_and_fetch(
            client, child_table, parent_table,
            child_df, child_vecs, parent_lookup,
            query, top_k=3,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        print_results(results, query)
        print(f"\n  ⏱ {elapsed:.2f}s")

    print(f"\n{'='*65}")
    print(f"[DONE] {len(TEST_QUERIES)} queries, avg {total_time/len(TEST_QUERIES):.2f}s each")
    print(f"{'='*65}")



if __name__ == "__main__":
    main()
