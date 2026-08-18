"""Test 5.3.1 section lookup fix."""
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
import numpy as np, lancedb
from google import genai
from src.retrieval import PhysicsTutorRetriever, build_context

GCP_PROJECT  = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

client   = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
db       = lancedb.connect("data/lancedb")
child_df = db.open_table("child_chunks").to_pandas()
cvecs    = np.array(child_df["vector"].tolist(), dtype=np.float32)
parent_df= db.open_table("parent_chunks").to_pandas()
retriever= PhysicsTutorRetriever(child_df, cvecs, parent_df)

q   = "chatpter 5 er 5.3.1 ta bujhi nai"
emb = client.models.embed_content(model=EMBED_MODEL, contents=[q])
qv  = list(emb.embeddings[0].values)
results, hints = retriever.retrieve(q, qv, top_k=5)

print(f"Intent={hints.intent.value} ch={hints.chapter} sec={hints.section}")
print(f"Retrieved: {len(results)} chunks")
for r in results:
    ctype = r["content_type"]
    ps = r["page_start"]
    pe = r["page_end"]
    txt = r["text"][:70]
    print(f"  [{ctype}] pg={ps}-{pe} | {txt}")

print()
ctx = build_context(results, hints)
# Check if Archimedes content is in context
keywords = ["আর্কিমিডিস", "প্লবতা", "Archimedes", "buoyan", "অপসারিত"]
found = [kw for kw in keywords if kw.lower() in ctx.lower()]
print(f"Archimedes keywords in context: {found if found else 'NOT FOUND ✗'}")
