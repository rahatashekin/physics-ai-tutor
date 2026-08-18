"""Check what retrieval returns for Snell's law query and figure query."""
import io, sys, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
import lancedb
from dotenv import load_dotenv; load_dotenv()
from src.retrieval import PhysicsTutorRetriever, build_context, extract_keywords
from src.figure_context import get_figures_for_pages, figures_to_context

db = lancedb.connect('data/lancedb')
pdf = db.open_table('parent_chunks').to_pandas()
cdf = db.open_table('child_chunks').to_pandas()
cvec = np.array(cdf['vector'].tolist(), dtype=np.float32)
r = PhysicsTutorRetriever(cdf, cvec, pdf)

import vertexai, os
from vertexai.language_models import TextEmbeddingModel
vertexai.init(project=os.getenv("GCP_PROJECT"), location="us-central1")
em = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

q_snell = 'স্নেলের সূত্র কী? সূত্রটা বুঝিয়ে দাও।'
q_fig   = 'আলোর পূর্ণ অভ্যন্তরীণ প্রতিফলনের চিত্র দেখাও।'

for q in [q_snell, q_fig]:
    print(f'\n=== Query: {q[:55]} ===')
    vec = em.get_embeddings([q])[0].values
    results, hints = r.retrieve(q, vec, top_k=5)
    print(f'  Intent={hints.intent}, Ch={hints.chapter}')
    for i, res in enumerate(results[:4]):
        pg = res.get('page_start')
        ch = res.get('chapter')
        ct = res.get('content_type', '')
        sc = res.get('score', 0)
        print(f'  [{i}] ch={ch} pg={pg} [{ct}] score={sc:.2f}: {str(res.get("text",""))[:80]}')
    
    # Check figures
    figs = get_figures_for_pages(results[0].get('page_start',0)-2, results[0].get('page_end',0)+2)
    print(f'  Figures in page range: {len(figs)}')
    for fig in figs:
        desc = fig.get('description','')[:60]
        print(f'    pg={fig.get("book_page")} key={fig.get("key","")}: {desc}')
