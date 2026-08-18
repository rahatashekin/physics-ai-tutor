"""Test BookIndex-based retrieval across multiple chapters."""
import io, sys, os
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

TEST_QUERIES = [
    # Previously MISSING chapters
    "chapter 5 এর নমুনা প্রশ্নের ১ নম্বরটা কী?",
    "chapter 8 এর সৃজনশীল প্রশ্ন ১ টা কী?",
    "chapter 4 এর সংক্ষিপ্ত উত্তর প্রশ্ন গুলো কী কী?",
    "chapter 10 এর নমুনা প্রশ্ন দাও",
    # ch2 should still work
    "chapter 2 এর নমুনা প্রশ্ন ৩ কী?",
]

def main():
    print("=== Cross-Chapter Exercise Retrieval Test ===\n")
    genai_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    oai_client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    db           = lancedb.connect("data/lancedb")
    child_df     = db.open_table("child_chunks").to_pandas()
    child_vecs   = np.array(child_df["vector"].tolist(), dtype=np.float32)
    parent_df    = db.open_table("parent_chunks").to_pandas()
    retriever    = PhysicsTutorRetriever(child_df, child_vecs, parent_df)

    for q in TEST_QUERIES:
        print(f"\nQ: {q}")
        emb   = genai_client.models.embed_content(model=EMBED_MODEL, contents=[q])
        q_vec = list(emb.embeddings[0].values)
        results, hints = retriever.retrieve(q, q_vec, top_k=3)

        print(f"  Intent={hints.intent.value} ch={hints.chapter} q#={hints.question_num}")
        if results:
            r = results[0]
            print(f"  Source: [{r['content_type']}] chunk_id={r['chunk_id']}")
            print(f"  Text preview: {r['text'][:120]}")
        else:
            print("  ✗ NO RESULTS")

        # Quick LLM answer
        ctx  = build_context(results, hints)
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "তুমি পদার্থবিজ্ঞান tutor। বাংলায় সংক্ষিপ্ত উত্তর দাও।"},
                {"role": "user",   "content": f"প্রশ্ন: {q}\n\nContent:\n{ctx[:1500]}\n\nউত্তর দাও।"},
            ],
            max_tokens=250, temperature=0.2,
        )
        print(f"  LLM: {resp.choices[0].message.content[:150]}")

main()
