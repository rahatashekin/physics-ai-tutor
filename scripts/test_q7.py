"""Quick test for Q7 only."""
import io, sys, os
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np
import lancedb
from google import genai
from openai import OpenAI
from src.retrieval import PhysicsTutorRetriever, build_context, extract_hints

GCP_PROJECT  = os.getenv("GCP_PROJECT", "project-3e580a5b-256c-4c6d-a0a")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

def main():
    genai_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    oai_client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    db           = lancedb.connect("data/lancedb")
    child_df     = db.open_table("child_chunks").to_pandas()
    child_vecs   = np.array(child_df["vector"].tolist(), dtype=np.float32)
    parent_df    = db.open_table("parent_chunks").to_pandas()
    retriever    = PhysicsTutorRetriever(child_df, child_vecs, parent_df)

    q = "chapter 2 এর নমুনা প্রশ্ন ২ এবং সংক্ষিপ্ত প্রশ্ন ৩ কী কী? সমাধান করো।"
    hints = extract_hints(q)
    print(f"Intent: {hints.intent.value}, ch={hints.chapter}")

    emb   = genai_client.models.embed_content(model=EMBED_MODEL, contents=[q])
    q_vec = list(emb.embeddings[0].values)
    results, hints2 = retriever.retrieve(q, q_vec, top_k=5)

    print(f"Retrieved: {len(results)} chunks")
    for r in results:
        print(f"  [{r['content_type']}] pg={r['page_start']}-{r['page_end']} | {r['text'][:60]}")

    ctx = build_context(results, hints2)
    
    SYSTEM = "তুমি একজন পদার্থবিজ্ঞান tutor। সবসময় বাংলায় উত্তর দাও। বইয়ের chapter/page উল্লেখ করো।"
    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"প্রশ্ন: {q}\n\nবইয়ের content:\n{ctx}\n\nউত্তর দাও।"},
        ],
        max_tokens=600, temperature=0.3,
    )
    print("\nANSWER:")
    print(resp.choices[0].message.content)

main()
