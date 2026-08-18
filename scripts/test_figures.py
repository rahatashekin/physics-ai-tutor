"""Test figure description integration."""
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
from src.figure_context import get_figures_for_pages, figures_to_context

GCP_PROJECT  = os.getenv("GCP_PROJECT", "project-3e580a5b-256c-4c6d-a0a")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

def test_figure_lookup():
    print("=== Figure Lookup Test ===\n")
    # chapter 2 page 43 এ figure আছে?
    figs = get_figures_for_pages(43, 44)
    print(f"Pages 43-44: {len(figs)} figures")
    for f in figs:
        print(f"  pg={f['book_page']} key={f['key']}")
        print(f"  desc: {f['description'][:120]}")
        print()

    # chapter 2 page 36 figure
    figs2 = get_figures_for_pages(36, 36)
    print(f"Pages 36: {len(figs2)} figures")
    for f in figs2:
        print(f"  pg={f['book_page']} desc: {f['description'][:100]}")


def test_figure_in_context():
    print("\n=== Figure in Context Test ===\n")
    
    genai_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    oai_client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    db           = lancedb.connect("data/lancedb")
    child_df     = db.open_table("child_chunks").to_pandas()
    child_vecs   = np.array(child_df["vector"].tolist(), dtype=np.float32)
    parent_df    = db.open_table("parent_chunks").to_pandas()
    retriever    = PhysicsTutorRetriever(child_df, child_vecs, parent_df)

    # Q: page 43 এ যে চিত্র আছে সেটা কী দেখাচ্ছে?
    q = "chapter 2 এর section 2.6 এ যে চিত্র আছে সেটা কী দেখাচ্ছে? চিত্রটা বুঝিয়ে দাও।"
    print(f"Q: {q}\n")

    emb   = genai_client.models.embed_content(model=EMBED_MODEL, contents=[q])
    q_vec = list(emb.embeddings[0].values)
    results, hints = retriever.retrieve(q, q_vec, top_k=4)

    print(f"Intent: {hints.intent.value}, sec={hints.section}")
    print(f"Retrieved: {len(results)} chunks")
    for r in results:
        print(f"  [{r['content_type']}] pg={r['page_start']}-{r['page_end']}")

    ctx = build_context(results, hints)
    print(f"\nContext length: {len(ctx)} chars")
    
    # Check if figure injection happened
    if "সংশ্লিষ্ট চিত্রের বর্ণনা" in ctx:
        print("✓ Figure descriptions injected!")
        fig_part = ctx.split("সংশ্লিষ্ট চিত্রের বর্ণনা")[1]
        print(f"  Figure text: {fig_part[:200]}")
    else:
        print("✗ No figure descriptions found for this page range")

    SYSTEM = "তুমি একজন পদার্থবিজ্ঞান tutor। চিত্রের বর্ণনা দেওয়া থাকলে সেটা ব্যবহার করে explain করো। সবসময় বাংলায় উত্তর দাও।"
    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"প্রশ্ন: {q}\n\nবইয়ের content (চিত্রসহ):\n{ctx}\n\nউত্তর দাও।"},
        ],
        max_tokens=500, temperature=0.3,
    )
    print("\nANSWER:")
    print(resp.choices[0].message.content)


test_figure_lookup()
test_figure_in_context()
