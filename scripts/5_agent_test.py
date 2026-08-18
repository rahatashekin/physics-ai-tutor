"""
Full system test — 8 question types × multiple chapters.
Each category tested with 2-3 different chapters.
"""
import io, sys, os, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, lancedb
from google import genai
from openai import OpenAI
from src.retrieval import PhysicsTutorRetriever, build_context, extract_hints

GCP_PROJECT  = os.getenv("GCP_PROJECT", "project-3e580a5b-256c-4c6d-a0a")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

SYSTEM = """তুমি একজন পদার্থবিজ্ঞান tutor।
- সবসময় বাংলায় উত্তর দাও
- বইয়ের chapter/page/section উল্লেখ করো
- সংক্ষিপ্ত কিন্তু সম্পূর্ণ উত্তর দাও"""

# ─── Test cases: (label, question) ─────────────────────────────────
TESTS = [
    # ── TYPE 1: Chapter Overview ──────────────────────────────────
    ("OVERVIEW ch3",  "chapter 3 এর সব sections এর নাম ও কী আছে সংক্ষেপে বলো"),
    ("OVERVIEW ch8",  "chapter 8 (আলো) এ কী কী বিষয় আছে? overview দাও"),

    # ── TYPE 2: Section Lookup ────────────────────────────────────
    ("SECTION 3.4",   "chapter 3 এর section 3.4 কী নিয়ে? বুঝিয়ে দাও"),
    ("SECTION 6.3",   "chapter 6 এর 6.3 section বুঝতে পারছি না"),

    # ── TYPE 3: Page Lookup ───────────────────────────────────────
    ("PAGE ch5 p110", "page 110 এ কী আছে?"),
    ("PAGE ch8 p200", "page 200 এ কী পড়ানো হয়েছে?"),

    # ── TYPE 4: নিজে করো ─────────────────────────────────────────
    ("NICHE_KORO ch3", "chapter 3 এ কোনো নিজে করো অংশ আছে? সেটা কী?"),
    ("NICHE_KORO ch7", "chapter 7 এর নিজে করো অংশটা বুঝিয়ে দাও"),

    # ── TYPE 5: অনুসন্ধান ────────────────────────────────────────
    ("ONUSONDAN ch2",  "chapter 2 এর অনুসন্ধান ২.০৩ কী? কয়টা page জুড়ে?"),
    ("ONUSONDAN ch4",  "chapter 4 এর অনুসন্ধান কোনটি? বর্ণনা দাও"),

    # ── TYPE 6: নমুনা প্রশ্ন ─────────────────────────────────────
    ("MCQ ch5 q1",    "chapter 5 এর নমুনা প্রশ্ন ১ নম্বরটা কী? উত্তর বলো"),
    ("MCQ ch7 q3",    "chapter 7 এর বহুনির্বাচনি প্রশ্ন ৩ নম্বরটা কী?"),
    ("MCQ ch11",      "chapter 11 এর নমুনা প্রশ্ন দাও"),

    # ── TYPE 7: সংক্ষিপ্ত প্রশ্ন ────────────────────────────────
    ("SHORT ch6",     "chapter 6 এর সংক্ষিপ্ত উত্তর প্রশ্নগুলো কী কী?"),
    ("SHORT ch9 q2",  "chapter 9 এর সংক্ষিপ্ত প্রশ্ন ২ নম্বরটা কী?"),
    ("SHORT ch13",    "chapter 13 এর সংক্ষিপ্ত প্রশ্ন দাও"),

    # ── TYPE 8: সৃজনশীল প্রশ্ন ──────────────────────────────────
    ("CREATIVE ch4",  "chapter 4 এর সৃজনশীল প্রশ্ন ১ টা কী?"),
    ("CREATIVE ch12", "chapter 12 এর সৃজনশীল প্রশ্ন দাও"),
]

PASS = "✅"
FAIL = "❌"

def main():
    print("=" * 70)
    print("Full System Test — 8 Question Types × Multiple Chapters")
    print("=" * 70)

    genai_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    oai_client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    db           = lancedb.connect("data/lancedb")
    child_df     = db.open_table("child_chunks").to_pandas()
    child_vecs   = np.array(child_df["vector"].tolist(), dtype=np.float32)
    parent_df    = db.open_table("parent_chunks").to_pandas()
    retriever    = PhysicsTutorRetriever(child_df, child_vecs, parent_df)

    print(f"DB: {len(child_df)} child, {len(parent_df)} parent chunks\n")

    results_summary = []

    for label, q in TESTS:
        print(f"\n{'─'*70}")
        print(f"[{label}]")
        print(f"Q: {q}")

        t0 = time.time()

        # Embed
        emb   = genai_client.models.embed_content(model=EMBED_MODEL, contents=[q])
        q_vec = list(emb.embeddings[0].values)

        # Retrieve
        results, hints = retriever.retrieve(q, q_vec, top_k=4)
        elapsed_r = time.time() - t0

        print(f"  Intent={hints.intent.value} ch={hints.chapter} pg={hints.page} sec={hints.section}")
        retrieved = len(results) > 0
        if results:
            r = results[0]
            src_tag = f"[{r['content_type']}] chunk={r['chunk_id'][:25]}"
            print(f"  Retrieved: {len(results)} chunks | Top: {src_tag}")
            print(f"  Text: {r['text'][:80]}")
        else:
            print(f"  {FAIL} NO RESULTS")

        # LLM answer
        ctx  = build_context(results, hints)
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": f"প্রশ্ন: {q}\n\nContent:\n{ctx[:2000]}\n\nউত্তর দাও।"},
            ],
            max_tokens=200, temperature=0.2,
        )
        answer  = resp.choices[0].message.content.strip()
        elapsed = time.time() - t0

        # Heuristic pass check: answer has Bengali text and is not empty/apology
        fail_phrases = ["পাওয়া যায়নি", "উত্তর দিতে পারছি না", "তথ্য নেই", "no content"]
        passed = retrieved and len(answer) > 30 and not any(p in answer for p in fail_phrases)

        status = PASS if passed else FAIL
        print(f"  {status} ({elapsed:.1f}s) Answer: {answer[:120]}")
        results_summary.append((label, status, elapsed))

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    passed_n = sum(1 for _, s, _ in results_summary if s == PASS)
    total_n  = len(results_summary)
    print(f"Score: {passed_n}/{total_n}\n")

    for label, status, t in results_summary:
        print(f"  {status} [{label}] ({t:.1f}s)")

    if passed_n == total_n:
        print(f"\n🎉 সব {total_n}টা test pass! System ready.")
    else:
        failed = [l for l, s, _ in results_summary if s == FAIL]
        print(f"\n⚠️  Failed: {failed}")


if __name__ == "__main__":
    main()
